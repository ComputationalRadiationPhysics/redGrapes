/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/resource/user_list.hpp
 */

#pragma once

#ifndef RESOURCE_USER_LIST_CHUNK_SIZE
#    define RESOURCE_USER_LIST_CHUNK_SIZE 1024
#endif

namespace redGrapes
{
    struct ChunkedList<T, chunk_size = 1024>
    {
        std::list<std::array<T*, chunk_size>, memory::Allocator<std::array<T, chunk_size>>> chunks;

        struct BackwardIterator
        {
        };

        BackwardIterator push(T&& item)
        {

            
            
            return BackwardIterator{
                
            };
        }
    };

} // namespace redGrapes
