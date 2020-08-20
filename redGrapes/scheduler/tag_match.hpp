/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <bitset>
#include <optional>
#include <memory>
#include <redGrapes/scheduler/scheduler.hpp>

namespace redGrapes
{
namespace scheduler
{

template < std::size_t T_tag_count = 64 >
struct SchedulingTagProperties
{
    std::bitset< T_tag_count > required_scheduler_tags;

    template < typename PropertiesBuilder >
    struct Builder
    {
        PropertiesBuilder & builder;

        Builder( PropertiesBuilder & b )
            : builder(b)
        {}

        PropertiesBuilder scheduling_tags( std::initializer_list< unsigned > tags )
        {
            std::bitset< T_tag_count > tags_bitset;
            for( auto tag : tags )
                tags_bitset.set( tag );
            return scheduling_tags( tags_bitset );
        }

        PropertiesBuilder scheduling_tags( std::bitset< T_tag_count > tags )
        {
            builder.prop.required_scheduler_tags |= tags;
            return builder;
        }
    };
};

template <
    typename TaskID,
    typename TaskPtr,
    std::size_t T_tag_count = 64
>
struct TagMatch : IScheduler< TaskID, TaskPtr >
{
    struct SubScheduler
    {
        std::bitset< T_tag_count > supported_tags;
        std::shared_ptr< IScheduler< TaskID, TaskPtr > > s;
    };

    std::vector< SubScheduler > sub_schedulers;

    void add_scheduler(
        std::bitset< T_tag_count > supported_tags,
        std::shared_ptr< IScheduler< TaskID, TaskPtr > > s
    )
    {
        sub_schedulers.push_back(
            SubScheduler
            {
                supported_tags,
                s
            });
    }

    void add_scheduler(
        std::initializer_list< unsigned > tag_list,
        std::shared_ptr< IScheduler< TaskID, TaskPtr > > s                       
    )
    {
        std::bitset< T_tag_count > supported_tags;
        for( auto tag : tag_list )
            supported_tags.set( tag );
        this->add_scheduler( supported_tags, s );
    }
    
    void init_mgr_callbacks(
        std::shared_ptr< redGrapes::SchedulingGraph< TaskID, TaskPtr > > scheduling_graph,
        std::function< bool ( TaskPtr ) > run_task,
        std::function< void ( TaskPtr ) > activate_followers,
        std::function< void ( TaskPtr ) > remove_task
    )
    {
        for( auto & s : sub_schedulers )
            s.s->init_mgr_callbacks( scheduling_graph, run_task, activate_followers, remove_task );
    }

    void notify()
    {
        for( auto & s : sub_schedulers )
            s.s->notify();
    }

    std::optional<
        std::shared_ptr< IScheduler< TaskID, TaskPtr > >
    >
    get_matching_scheduler(
        std::bitset< T_tag_count > const & required_tags
    )
    {
        for( auto const & s : sub_schedulers )
            if( ( s.supported_tags & required_tags ) == required_tags )
                return s.s;

        return std::nullopt;
    }

    bool
    task_dependency_type(
        TaskPtr a,
        TaskPtr b
    )
    {
        /// fixme: b or a ?
        if( auto sub_scheduler = get_matching_scheduler( b.get().required_scheduler_tags ) )
            return (*sub_scheduler)->task_dependency_type( a, b );
        else
            throw std::runtime_error("no scheduler found for task");
    }

    void
    activate_task( TaskPtr task_ptr )
    {
        if( auto sub_scheduler = get_matching_scheduler( task_ptr.get().required_scheduler_tags ) )
            return (*sub_scheduler)->activate_task( task_ptr );
        else
            throw std::runtime_error("no scheduler found for task");
    }
};

/*! Factory function to easily create a tag-match-scheduler object
 */
template < typename Manager >
struct TagMatchBuilder
{
    std::shared_ptr< TagMatch< typename Manager::TaskID, typename Manager::TaskPtr > > tag_match;

    operator std::shared_ptr< IScheduler< typename Manager::TaskID, typename Manager::TaskPtr > > () const
    {
        return tag_match;
    }

    TagMatchBuilder add( std::initializer_list<unsigned> tags, std::shared_ptr< IScheduler< typename Manager::TaskID, typename Manager::TaskPtr > > s )
    {
        tag_match->add_scheduler( tags, s );
        return *this;
    }
};

template <
    typename Manager,
    size_t T_tag_count = 64
>
auto make_tag_match_scheduler(
    Manager & m
)
{
    return TagMatchBuilder< Manager > {
        std::make_shared<
               TagMatch<
                   typename Manager::TaskID,
                   typename Manager::TaskPtr,
                   T_tag_count
               >
        >()
    };
}


} // namespace scheduler

} // namespace redGrapes

