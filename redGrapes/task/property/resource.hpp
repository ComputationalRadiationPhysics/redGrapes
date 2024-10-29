/* Copyright 2019-2024 Michael Sippel, Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/property/resource.hpp
 */

#pragma once

#include "redGrapes/TaskFreeCtx.hpp"
#include "redGrapes/globalSpace.hpp"
#include "redGrapes/resource/resource_user.hpp"

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <list>
#include <stdexcept>

namespace redGrapes
{

    template<typename TTask>
    struct ResourceProperty : ResourceUser
    {
        ResourceProperty(WorkerId worker_id, unsigned scope_depth) : ResourceUser(worker_id, scope_depth)
        {
        }

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

            inline PropertiesBuilder& add_resource(ResourceAccess const& access)
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

        // Only to be called inside a running task to update property
        void apply_patch(Patch const& patch)
        {
            ResourceUser before(*this, static_cast<TTask*>(current_task)->worker_id);

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

} // namespace redGrapes

template<typename TTask>
struct fmt::formatter<redGrapes::ResourceProperty<TTask>>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(redGrapes::ResourceProperty<TTask> const& label_prop, FormatContext& ctx) const
    {
        return fmt::format_to(ctx.out(), "\"resources\" : {}", (redGrapes::ResourceUser const&) label_prop);
    }
};
