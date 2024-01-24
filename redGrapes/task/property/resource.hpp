/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/property/resource.hpp
 */

#pragma once

#include <redGrapes/resource/resource.hpp>
#include <redGrapes/resource/resource_user.hpp>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <cstdarg>
#include <stdexcept>

namespace redGrapes
{

    struct ResourceProperty : ResourceUser
    {
        template<typename PropertiesBuilder>
        struct Builder
        {
            PropertiesBuilder& builder;

            Builder(PropertiesBuilder& b) : builder(b)
            {
            }

            PropertiesBuilder& resources(std::initializer_list<ResourceAccess> list)
            {
                for(ResourceAccess const& ra : list)
                    builder.task->access_list.push(ra);
                builder.task->build_unique_resource_list();

                return builder;
            }

            inline PropertiesBuilder& add_resource(ResourceAccess access)
            {
                (*builder.task) += access;
                return builder;
            }
        };

        struct Patch
        {
            template<typename PatchBuilder>
            struct Builder
            {
                PatchBuilder& builder;

                Builder(PatchBuilder& b) : builder(b)
                {
                }

                PatchBuilder add_resources(std::initializer_list<ResourceAccess> list)
                {
                    Patch& p = builder.patch;
                    for(auto const& acc : list)
                        p += acc;
                    return builder;
                }

                PatchBuilder remove_resources(std::initializer_list<ResourceAccess> list)
                {
                    Patch& p = builder.patch;
                    for(auto const& acc : list)
                        p -= acc;
                    return builder;
                }
            };

            enum DiffType
            {
                ADD,
                REMOVE
            };

            std::list<std::pair<DiffType, ResourceAccess>> diff;

            void operator+=(Patch const& other)
            {
                this->diff.insert(std::end(this->diff), std::begin(other.diff), std::end(other.diff));
            }

            void operator+=(ResourceAccess const& ra)
            {
                this->diff.push_back(std::make_pair(DiffType::ADD, ra));
            }

            void operator-=(ResourceAccess const& ra)
            {
                this->diff.push_back(std::make_pair(DiffType::REMOVE, ra));
            }
        };

        inline void operator+=(ResourceAccess const& ra)
        {
            this->add_resource_access(ra);
        }

        inline void operator-=(ResourceAccess const& ra)
        {
            this->rm_resource_access(ra);
        }

        void apply_patch(Patch const& patch)
        {
            ResourceUser before(*this);

            for(auto x : patch.diff)
            {
                switch(x.first)
                {
                case Patch::DiffType::ADD:
                    (*this) += x.second;
                    break;
                case Patch::DiffType::REMOVE:
                    (*this) -= x.second;
                    break;
                }
            }

            if(!before.is_superset_of(*this))
                throw std::runtime_error("redGrapes: ResourceUserPolicy: updated access list is no subset!");
        }
    };

    struct ResourcePrecedencePolicy
    {
        static bool is_serial(ResourceProperty const& a, ResourceProperty const& b)
        {
            return redGrapes::ResourceUser::is_serial(a, b);
        }

        static void assert_superset(ResourceProperty const& super, ResourceProperty const& sub)
        {
            if(!redGrapes::ResourceUser::is_superset(super, sub))
            {
                auto msg = fmt::format("Not allowed: {} is no superset of {}\n", super, sub);
                spdlog::error(msg);
                throw std::runtime_error(msg);
            }
        }
    };

} // namespace redGrapes

template<>
struct fmt::formatter<redGrapes::ResourceProperty>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(redGrapes::ResourceProperty const& label_prop, FormatContext& ctx)
    {
        return format_to(ctx.out(), "\"resources\" : {}", (redGrapes::ResourceUser const&) label_prop);
    }
};
