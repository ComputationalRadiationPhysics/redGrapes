/* Copyright 2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/property/id.hpp
 */

#pragma once

#include <fmt/format.h>

#include <atomic>

namespace redGrapes
{

    using TaskID = unsigned int;

    struct IDProperty
    {
    private:
        static std::atomic_int& id_counter()
        {
            static std::atomic_int x;
            return x;
        }

    public:
        TaskID task_id;

        IDProperty() : task_id(-1) // id_counter().fetch_add( 1, std::memory_order_seq_cst ) )
        {
        }

        IDProperty(IDProperty&& other) : task_id(other.task_id)
        {
        }

        IDProperty(IDProperty const& other) : task_id(other.task_id)
        {
        }

        IDProperty& operator=(IDProperty const& other)
        {
            return *this;
        }

        template<typename PropertiesBuilder>
        struct Builder
        {
            PropertiesBuilder& b;

            Builder(PropertiesBuilder& b) : b(b)
            {
            }

            void init_id()
            {
                b.task->task_id = id_counter().fetch_add(1, std::memory_order_seq_cst);
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

        void apply_patch(Patch const&){};
    };

} // namespace redGrapes

template<>
struct fmt::formatter<redGrapes::IDProperty>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(redGrapes::IDProperty const& id_prop, FormatContext& ctx)
    {
        return format_to(ctx.out(), "\"id\" : {}", id_prop.task_id);
    }
};
