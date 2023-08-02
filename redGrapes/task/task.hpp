/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <type_traits>
#include <redGrapes/task/task_base.hpp>
#include <redGrapes/task/property/inherit.hpp>
#include <redGrapes/task/property/trait.hpp>
#include <redGrapes/task/property/id.hpp>
#include <redGrapes/task/property/resource.hpp>
#include <redGrapes/task/property/queue.hpp>
#include <redGrapes/task/property/graph.hpp>
#include <redGrapes/util/trace.hpp>


// defines REDGRAPES_TASK_PROPERTIES
#include <redGrapes_config.hpp>

namespace redGrapes
{

using TaskProperties = TaskProperties1<
    IDProperty,
    ResourceProperty,
    QueueProperty,
    GraphProperty
#ifdef REDGRAPES_TASK_PROPERTIES
    , REDGRAPES_TASK_PROPERTIES
#endif
>;

struct Task :
        TaskBase,
        TaskProperties,
        std::enable_shared_from_this<Task>
{
    virtual ~Task() {}

    unsigned arena_id;
    std::atomic_int removal_countdown;

    Task()
        : removal_countdown(2)
    {}

    virtual void * get_result_data()
    {
        return nullptr;
    }
};

// TODO: fuse ResultTask and FunTask into one template
//     ---> removes one layer of virtual function calls

template < typename Result >
struct ResultTask : Task
{
    Result result_data;

    virtual ~ResultTask() {}

    virtual void * get_result_data()
    {
        return &result_data;
    }

    virtual Result run_result() {}

    void run() final
    {
        result_data = run_result();
        get_result_set_event().notify(); // result event now ready
    }   
};

template<>
struct ResultTask<void> : Task
{
    virtual ~ResultTask() {}
 
    virtual void run_result() {}

    void run() final
    {
        run_result();
        get_result_set_event().notify();
    }
};

template< typename F >
struct FunTask
    : ResultTask< typename std::result_of<F()>::type >
{
    std::optional< F > impl;

    virtual ~FunTask() {}

    typename std::result_of<F()>::type run_result()
    {
        return (*this->impl)();
    }
};

} // namespace redGrapes

