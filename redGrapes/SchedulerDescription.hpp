/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <memory>

namespace redGrapes
{

    struct DefaultTag
    {
    };

    template<typename T>
    concept C_Exec = requires(T execDesc) {
        typename T::Key;
        typename T::ValueType;
        { execDesc.scheduler };
    };

    template<typename TScheduler, typename TTag>
    struct SchedulerDescription
    {
        using Key = TTag;
        using ValueType = TScheduler;

        SchedulerDescription(std::shared_ptr<TScheduler> const& scheduler, TTag = DefaultTag{}) : scheduler{scheduler}
        {
        }

        std::shared_ptr<TScheduler> scheduler;
    };

} // namespace redGrapes
