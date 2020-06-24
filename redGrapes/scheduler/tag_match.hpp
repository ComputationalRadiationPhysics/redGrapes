/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <bitset>
#include <optional>

namespace redGrapes
{
namespace scheduler
{

struct IScheduler
{
    virtual ~IScheduler() {};

    bool task_dependency_type( TaskPtr a, TaskPtr b );
    void activate_task( TaskPtr task_ptr );
};

template <
    std::size_t T_tag_count = 64
>
struct TagMatch : IScheduler
{
    struct SubScheduler
    {
        std::bitset< T_tag_count > supported_tags;
        std::bitset< T_tag_count > required_tags;

        std::shared_ptr< IScheduler > s;
    };

    std::vector< SubScheduler > sub_schedulers;

    struct TaskProperties
    {
        std::bitset< T_tag_count > required_scheduler_tags;

        template < typename PropertiesBuilder >
        struct Builder
        {
            PropertiesBuilder & builder;

            Builder( PropertiesBuilder & b )
                : builder(b)
            {}

            // fixme: return reference?
            PropertiesBuilder scheduler_tags( std::bitset< T_tag_count > tags )
            {
                required_scheduler_tags |= tags;
                return builder;
            }
        };
    };

    void add_scheduler(
        std::bitset< T_tag_count > supported_tags,
        std::bitset< T_tag_count > required_tags,
        std::shared_ptr< Scheduler > s
    )
    {
        sub_schedulers.push_back(
            SubScheduler
            {
                supported_tags,
                s
            });
    }

    std::optional<
        std::shared_ptr< SubScheduler >
    >
    get_matching_scheduler(
        std::bitset< T_tag_count > const & required_tags
    )
    {
        for( auto const & s : sub_schedulers )
        {
            if( ( s.supported_tags & required_tags ) == required_tags &&
                ( s.required_tags & required_tags ) == s.required_tags )
                return s.s;
        }

        return std::nullopt;
    }

    bool
    task_dependency_type(
        TaskPtr a,
        TaskPtr b
    )
    {
        /// fixme: b or a ?
        if( auto sub_scheduler = get_matching_scheduler( b.get_locked().required_scheduler_tags ) )
            (*sub_scheduler)->task_dependency_type( a, b );
        else
            throw std::runtime_error("no scheduler found for task");
    }

    void
    activate_task( TaskPtr task_ptr )
    {
        if( auto sub_scheduler = get_matching_scheduler( task_ptr.get_locked().required_scheduler_tags ) )
            (*sub_scheduler)->activate_task( task_ptr );
        else
            throw std::runtime_error("no scheduler found for task");
    }
};

} // namespace scheduler

} // namespace redGrapes

