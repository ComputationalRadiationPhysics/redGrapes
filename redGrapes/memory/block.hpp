/* Copyright 2023 The RedGrapes Community
 *
 * Authors: Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/memory/block.hpp
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace redGrapes
{
    namespace memory
    {

        struct Block
        {
            uintptr_t ptr;
            std::size_t len;

            inline operator void*() const
            {
                return reinterpret_cast<void*>(ptr);
            }

            inline bool operator==(Block const& other) const
            {
                return ptr == other.ptr && len == other.len;
            }

            inline operator bool() const
            {
                return (bool) ptr;
            }

            static inline Block null()
            {
                return Block{.ptr = 0, .len = 0};
            }
        };

    } // namespace memory

} // namespace redGrapes
