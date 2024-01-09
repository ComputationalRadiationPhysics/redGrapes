/* Copyright 2023 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <atomic>
#include <optional>
#include <vector>

namespace redGrapes
{

    struct AtomicBitfield
    {
        AtomicBitfield(size_t m_size) : m_size(m_size), state(ceil_div(m_size, bitfield_len))
        {
        }

        size_t size()
        {
            return m_size;
        }

#define __INDEX_CALC(i, j, k, m)                                                                                      \
    unsigned j = i / bitfield_len;                                                                                    \
    unsigned k = i % bitfield_len;                                                                                    \
    uint64_t m = (uint64_t) 1 << k;

        /* atomically update bit at index `idx`
         *
         * @return previous value
         */
        inline bool set(unsigned idx, bool new_value)
        {
            __INDEX_CALC(idx, chunk_idx, k, mask)
            unsigned old_val;

            switch(new_value)
            {
            case false:
                old_val = state[chunk_idx].fetch_and(~mask, std::memory_order_acquire);
                break;

            case true:
                old_val = state[chunk_idx].fetch_or(mask, std::memory_order_release);
                break;
            }

            return old_val & mask;
        }

        /* get current value of bit at `idx`
         */
        inline bool get(unsigned idx)
        {
            __INDEX_CALC(idx, chunk_idx, k, mask)
            return state[chunk_idx] & mask;
        }

        /* searches for a bit which is of state `expected_value`
         * and suffices the condition given by `f`.
         *
         * @param start_idx gives a initial position where
         * elements in the same chunk as `start_idx` are preferred over
         * elements from differening chunks
         * and elements following `start_idx` are preferred over preceding ones
         *
         * @return element given by `f(idx)` where `state[idx] == expected_value`
         */
        template<typename T, typename F>
        inline std::optional<T> probe_by_value(
            F&& f,
            bool expected_value,
            unsigned start_idx,
            bool exclude_start = true)
        {
            uint64_t start_field_idx = start_idx / bitfield_len;
            uint64_t first_mask = (uint64_t(-1) << (start_idx % bitfield_len));
            uint64_t second_mask = ~first_mask;

            /* probe second-half of current chunk
             */
            if(start_field_idx == state.size() - 1 && size() % bitfield_len != 0)
                second_mask &= (uint64_t(1) << (size() % bitfield_len)) - 1;

            if(exclude_start)
                second_mask &= ~(uint64_t(1) << (start_idx % bitfield_len));

            if(auto x = probe_chunk_by_value<T>(start_field_idx, second_mask, expected_value, f))
                return x;

            /* probe first half of current chunk
             */
            if(start_field_idx == state.size() - 1 && size() % bitfield_len != 0)
                first_mask &= (uint64_t(1) << (size() % bitfield_len)) - 1;
            if(auto x = probe_chunk_by_value<T>(start_field_idx, first_mask, expected_value, f))
                return x;

            /* probe remaining chunks
             */
            for(uint64_t b = 1; b < ceil_div(size(), bitfield_len); ++b)
            {
                uint64_t field_idx = (start_field_idx + b) % state.size();
                uint64_t mask = ~0;

                if(field_idx == state.size() - 1 && size() % bitfield_len != 0)
                    mask &= (uint64_t(1) << (size() % bitfield_len)) - 1;

                if(auto x = probe_chunk_by_value<T>(field_idx, mask, expected_value, f))
                    return x;
            }

            return std::nullopt;
        }


    private:
        // TODO: try different values, e.g. 8
        // to add hierarchy matching the NUMA architecture
        static constexpr uint64_t bitfield_len = 64;

        size_t m_size;
        std::vector<std::atomic<uint64_t>> state;

        /*! calculates  ceil( a / b )
         */
        static inline uint64_t ceil_div(uint64_t a, uint64_t b)
        {
            return (a + b - 1) / b;
        }

        // find index of first set bit
        // taken from https://graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightParallel
        static inline unsigned int first_one_idx(uint64_t v)
        {
            unsigned int c = 64; // c will be the number of zero bits on the right
            v &= -int64_t(v);
            if(v)
                c--;
            if(v & 0x0000'0000'FFFF'FFFF)
                c -= 32;
            if(v & 0x0000'FFFF'0000'FFFF)
                c -= 16;
            if(v & 0x00FF'00FF'00FF'00FF)
                c -= 8;
            if(v & 0x0F0F'0F0F'0F0F'0F0F)
                c -= 4;
            if(v & 0x3333'3333'3333'3333)
                c -= 2;
            if(v & 0x5555'5555'5555'5555)
                c -= 1;

            return c;
        }

        /* searches for a bit which is of state `expected_value`
         * and suffices the condition given by `f` in the chunk `j`.
         *
         * @return element given by `f(idx)` where `state[idx] == expected_value`
         */
        template<typename T, typename F>
        inline std::optional<T> probe_chunk_by_value(unsigned j, uint64_t mask, bool expected_value, F&& f)
        {
            while(true)
            {
                uint64_t field = expected_value ? uint64_t(state[j]) : ~uint64_t(state[j]);

                uint64_t masked_field = field & mask;
                if(masked_field == 0)
                    break;

                // find index of first worker
                unsigned int k = first_one_idx(masked_field);

                if(k < bitfield_len)
                {
                    unsigned int idx = j * bitfield_len + k;
                    // spdlog::info("find worker: j = {}, k = {}, idx= {}", j , k, idx);

                    if(std::optional<T> x = f(idx))
                        return x;

                    // dont check this worker again
                    mask &= ~(uint64_t(1) << k);
                }
            }

            return std::nullopt;
        }
    };

} // namespace redGrapes
