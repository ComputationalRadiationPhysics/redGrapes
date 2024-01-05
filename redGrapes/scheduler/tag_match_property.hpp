
#pragma once

#include <fmt/format.h>

#include <bitset>
#include <memory>
#include <optional>

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

                PropertiesBuilder& scheduling_tags(std::initializer_list<unsigned> tags)
                {
                    std::bitset<T_tag_count> tags_bitset;
                    for(auto tag : tags)
                        tags_bitset.set(tag);
                    return scheduling_tags(tags_bitset);
                }

                PropertiesBuilder& scheduling_tags(std::bitset<T_tag_count> tags)
                {
                    builder.task->required_scheduler_tags |= tags;
                    return builder;
                }
            };

            struct Patch
            {
                template<typename PatchBuilder>
                struct Builder
                {
                    Builder(PatchBuilder&)
                    {
                    }
                };
            };

            void apply_patch(Patch const&)
            {
            }
        };
    } // namespace scheduler
} // namespace redGrapes

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
