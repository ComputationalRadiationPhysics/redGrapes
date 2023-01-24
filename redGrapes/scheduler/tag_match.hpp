/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <bitset>
#include <memory>
#include <optional>
#include <spdlog/spdlog.h>
#include <redGrapes/util/allocator.hpp>

namespace redGrapes
{
    namespace scheduler
    {
        template<typename Tag, std::size_t T_tag_count = 64>
        struct SchedulingTagProperties
        {
            std::bitset<T_tag_count> required_scheduler_tags;

            template<typename PropertiesBuilder>
            struct Builder
            {
                PropertiesBuilder& builder;

                Builder(PropertiesBuilder& b) : builder(b)
                {
                }

                PropertiesBuilder & scheduling_tags(std::initializer_list<unsigned> tags)
                {
                    std::bitset<T_tag_count> tags_bitset;
                    for(auto tag : tags)
                        tags_bitset.set(tag);
                    return scheduling_tags(tags_bitset);
                }

                PropertiesBuilder & scheduling_tags(std::bitset<T_tag_count> tags)
                {
                    builder.task->required_scheduler_tags |= tags;
                    return builder;
                }
            };

            struct Patch
            {
                template <typename PatchBuilder>
                struct Builder
                {
                    Builder( PatchBuilder & ) {}
                };
            };

            void apply_patch( Patch const & ) {}
        };
    }
}

template<typename Tag, std::size_t T_tag_count>
struct fmt::formatter<redGrapes::scheduler::SchedulingTagProperties<Tag, T_tag_count>>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(redGrapes::scheduler::SchedulingTagProperties<Tag, T_tag_count> const& prop, FormatContext& ctx)
    {
        auto out = ctx.out();

        out = fmt::format_to(out, "\"schedulingTags\" : [");

        bool first = true;
        for(size_t i = 0; i < T_tag_count; ++i)
        {
            if(prop.required_scheduler_tags.test(i))
            {
                if(!first)
                    out = format_to(out, ", ");

                first = false;
                out = format_to(out, "{}", (Tag) i);
            }
        }

        out = fmt::format_to(out, "]");
        return out;
    }
};



#include <vector>
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

            void start()
            {
                for(auto const& s : sub_schedulers)
                    s.s->start();
            }

            void stop()
            {
                for(auto const& s : sub_schedulers)
                    s.s->stop();
            }

            bool schedule( dispatch::thread::WorkerThread & worker )
            {
                for( auto& s : sub_schedulers )
		  if( s.s->schedule( worker ) )
		    return true;

		return false;                
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

            bool task_dependency_type(Task const & a, Task const & b)
            {
                /// fixme: b or a ?
                if(auto sub_scheduler = get_matching_scheduler(b.required_scheduler_tags))
                    return (*sub_scheduler)->task_dependency_type(a, b);
                else
                    throw std::runtime_error("no scheduler found for task");
            }

            void wake_all_workers()
            {
                for(auto const& s : sub_schedulers)
                    s.s->wake_all_workers();
            }

            bool wake_one_worker()
            {
                for(auto const& s : sub_schedulers)
                    if( s.s->wake_one_worker() )
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
            return TagMatchBuilder<T_tag_count>{memory::alloc_shared<TagMatch<T_tag_count>>()};
        }


    } // namespace scheduler

} // namespace redGrapes

