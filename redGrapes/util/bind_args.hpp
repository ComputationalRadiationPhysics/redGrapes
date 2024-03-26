/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once
#include <utility>

template<typename Callable, typename... Args>
struct BindArgs
{
    inline auto operator()(Callable&& f, Args&&... args)
    {
        return std::move([f = std::move(f), args...]() mutable { return f(std::forward<Args>(args)...); });
    }
};
