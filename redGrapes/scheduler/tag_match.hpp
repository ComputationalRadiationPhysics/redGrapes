/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once


#include <vector>
#include <spdlog/spdlog.h>

#include <redGrapes/task/task.hpp>
#include <redGrapes/scheduler/tag_match_property.hpp>
#include <redGrapes/memory/chunked_bump_alloc.hpp>
#include <redGrapes/scheduler/scheduler.hpp>

namespace redGrapes
{
namespace scheduler
{

        template<std::size_t T_tag_count = 64>
        struct TagMatch : IScheduler
        {
            struct SubScheduler
            {
                std::bitset<T_tag_count> supported_tags;
                std::shared_ptr< IScheduler > s;
            };

            std::vector<SubScheduler> sub_schedulers;

            void add_scheduler(std::bitset<T_tag_count> supported_tags, std::shared_ptr<IScheduler> s)
            {
                sub_schedulers.push_back(SubScheduler{supported_tags, s});
            }

            void add_scheduler(std::initializer_list<unsigned> tag_list, std::shared_ptr<IScheduler> s)
            {
                std::bitset<T_tag_count> supported_tags;
                for(auto tag : tag_list)
                    supported_tags.set(tag);
                this->add_scheduler(supported_tags, s);
            }

            Task * steal_task( dispatch::thread::Worker & worker )
            {
                for( auto& s : sub_schedulers )
		  if( Task * t = s.s->steal_task( worker ) )
		    return t;

		return nullptr;                
            }

            void emplace_task( Task & task )
            {
                if(auto sub_scheduler = get_matching_scheduler(task.required_scheduler_tags))
                    return (*sub_scheduler)->emplace_task(task);
                else
                    throw std::runtime_error("no scheduler found for task");                
            }

            void activate_task(Task & task)
            {
                if(auto sub_scheduler = get_matching_scheduler(task.required_scheduler_tags))
                    return (*sub_scheduler)->activate_task(task);
                else
                    throw std::runtime_error("no scheduler found for task");
            }

            std::optional<std::shared_ptr<IScheduler>> get_matching_scheduler(
                std::bitset<T_tag_count> const& required_tags)
            {
                for(auto const& s : sub_schedulers)
                    if((s.supported_tags & required_tags) == required_tags)
                        return s.s;

                return std::nullopt;
            }

            bool task_dependency_type(Task const & a, Task & b)
            {
                /// fixme: b or a ?
                if(auto sub_scheduler = get_matching_scheduler(b.required_scheduler_tags))
                    return (*sub_scheduler)->task_dependency_type(a, b);
                else
                    throw std::runtime_error("no scheduler found for task");
            }

            void wake_all()
            {
                for(auto const& s : sub_schedulers)
                    s.s->wake_all();
            }

            bool wake( WakerId waker_id )
            {
                for(auto const& s : sub_schedulers)
                    if( s.s->wake( waker_id ) )
                        return true;

                return false;
            }
        };

        /*! Factory function to easily create a tag-match-scheduler object
         */
        template< std::size_t T_tag_count = 64 >
        struct TagMatchBuilder
        {
            std::shared_ptr<TagMatch<T_tag_count>> tag_match;

            operator std::shared_ptr<IScheduler>() const
            {
                return tag_match;
            }

            TagMatchBuilder add(std::initializer_list<unsigned> tags, std::shared_ptr<IScheduler> s)
            {
                tag_match->add_scheduler(tags, s);
                return *this;
            }
        };

        template<size_t T_tag_count = 64>
        auto make_tag_match_scheduler()
        {
            return TagMatchBuilder<T_tag_count>{std::make_shared<TagMatch<T_tag_count>>()};
        }


    } // namespace scheduler

} // namespace redGrapes

