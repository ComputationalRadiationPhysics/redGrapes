/* Copyright 2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <redGrapes/task/future.hpp>
#include <redGrapes/task/task.hpp>
#include <redGrapes/task/task_space.hpp>
#include <redGrapes/memory/allocator.hpp>
#include <spdlog/spdlog.h>
#include <type_traits>

namespace redGrapes
{

/* HELPERS */

std::shared_ptr<TaskSpace> current_task_space();
void update_active_task_spaces();
Task * schedule( dispatch::thread::WorkerThread & worker );

template<typename... Args>
static inline void pass(Args&&...)
{
}

template <typename B>
struct PropBuildHelper
{
    typename TaskProperties::Builder<B>& builder;

    template<typename T>
    inline int build(T const& x)
    {
        trait::BuildProperties<T>::build(builder, x);
        return 0;
    }

    void foo()
    {
    }
};

/* TASK BUILDER */

template < typename Callable, typename... Args >
struct TaskBuilder
    : TaskProperties::Builder< TaskBuilder<Callable, Args...> >
{
    struct BindArgs
    {
        inline auto operator() ( Callable&& f, Args&&... args )
        {
            return std::move([f=std::move(f), args...]() mutable {
                return f(std::forward<Args>(args)...);
            });
        }
    };

    using Impl = typename std::result_of< BindArgs(Callable, Args...) >::type;
    using Result = typename std::result_of< Callable(Args...)>::type;

    std::shared_ptr< TaskSpace > space;
    FunTask< Impl > * task;

    TaskBuilder( Callable&& f, Args&&... args )
        : TaskProperties::Builder< TaskBuilder >( *this )
        , space( current_task_space() )
    {
        // allocate
        redGrapes::memory::Allocator< FunTask<Impl> > alloc;
        task = alloc.allocate( 1 );

        if( ! task )
            throw std::runtime_error("out of memory");

        // construct task in-place
        new (task) FunTask< Impl >( );

        task->arena_id = memory::current_arena;

        // init properties from args
        PropBuildHelper<TaskBuilder> build_helper{ *this };
        pass(build_helper.template build<Args>(std::forward<Args>(args))...);
        build_helper.foo();

        // init id
        this->init_id();

        // set impl
        task->impl.emplace(BindArgs{}( std::move(f), std::forward<Args>(args)... ));
    }

    TaskBuilder( TaskBuilder & other )
        : TaskProperties::Builder< TaskBuilder >( *this )
        , space( other.space )
        , task( other.task )
    {
        other.task = nullptr;
    }

    TaskBuilder( TaskBuilder && other )
        : TaskProperties::Builder< TaskBuilder >( *this )
        , space( std::move(other.space) )
        , task( std::move(other.task) )
    {
        other.task = nullptr;
    }

    ~TaskBuilder()
    {
        if( task )
            submit();
    }

    TaskBuilder & enable_stack_switching()
    {
        task->enable_stack_switching = true;
        return *this;
    }

    auto submit()
    {
        Task * t = task;
        task = nullptr;

        SPDLOG_TRACE("submit task {}", (TaskProperties const &)*t);
        space->submit( t );
        
        return std::move(Future<Result>( *t ));
    }

    auto get()
    {
        return submit().get();
    }
};

} // namespace redGrapes

