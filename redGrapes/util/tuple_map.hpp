/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <boost/mp11/map.hpp>
#include <boost/mp11/utility.hpp>

#include <memory>

namespace redGrapes
{
    namespace detail
    {
        template<typename T_Sequence, template<typename> typename T_Accessor = boost::mp11::mp_identity_t>
        struct InheritLinearly;

        template<typename... Ts, template<typename> typename T_Accessor>
        struct InheritLinearly<boost::mp11::mp_list<Ts...>, T_Accessor> : T_Accessor<Ts>...
        {
        };

        /** wrap a datum
         *
         * @tparam T_Pair mp_list<key, valueType>
         */

        template<typename T_Pair>
        struct TaggedValue
        {
            using Key = boost::mp11::mp_first<T_Pair>;
            using ValueType = boost::mp11::mp_second<T_Pair>;

            ValueType value;
        };
    } // namespace detail

    /** wrap a datum
     *
     * @tparam T_Map mp_list<pair>, where pair is mp_list< key, type of the value >
     */

    template<typename T_Map>
    struct MapTuple : protected detail::InheritLinearly<T_Map, detail::TaggedValue>
    {
        template<typename T_Key>
        using TaggedValueFor = detail::TaggedValue<boost::mp11::mp_map_find<T_Map, T_Key>>;

        /** access a value with a key
         *
         * @tparam T_Key key type
         *
         */
        template<typename T_Key>
        auto& operator[](T_Key const& key)
        {
            return static_cast<TaggedValueFor<T_Key>&>(*this).value;
        }

        template<typename T_Key>
        auto const& operator[](T_Key const& key) const
        {
            return static_cast<TaggedValueFor<T_Key>&>(*this).value;
        }

        template<typename Func>
        void apply_to_all(Func func)
        {
            // Iterate over each element of the map
            mp_for_each<MapTuple>(
                [&func](auto taggedValue)
                {
                    // Apply the function to the value of each element
                    func(taggedValue.value);
                });
        }

        template<typename MemberFunc, typename... Args>
        void call_member_func_for_all(MemberFunc memFunc, Args&&... args)
        {
            // Iterate over each element of the map
            mp_for_each<MapTuple>(
                [&memFunc, &args...](auto taggedValue)
                {
                    // Call the member function for each element
                    (taggedValue.value.*memFunc)(std::forward<Args>(args)...);
                });
        }
    };

    template<typename TDictObj>
    using MakeKeyValList = boost::mp11::mp_list<typename TDictObj::Key, std::shared_ptr<typename TDictObj::ValueType>>;


} // namespace redGrapes
