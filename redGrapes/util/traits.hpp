/* Copyright 2024 Tapish Narwal
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <type_traits>
#include <utility>

namespace redGrapes::traits
{
    // Primary template for is_specialization_of
    template<typename, template<typename...> class>
    struct is_specialization_of : std::false_type
    {
    };

    // Specialization for types that are specializations of the template
    template<typename... Args, template<typename...> class Template>
    struct is_specialization_of<Template<Args...>, Template> : std::true_type
    {
    };

    // Variable template for is_specialization_of
    template<typename T, template<typename...> class Template>
    inline constexpr bool is_specialization_of_v = is_specialization_of<T, Template>::value;

    template<typename T = std::false_type, typename...>
    struct first_type
    {
        using type = T;
    };

    // Convenience alias template
    template<typename... Ts>
    using first_type_t = typename first_type<Ts...>::type;

    template<typename T>
    struct is_pair : std::false_type
    {
    };

    template<typename T, typename U>
    struct is_pair<std::pair<T, U>> : std::true_type
    {
    };

    template<typename T>
    inline constexpr bool is_pair_v = is_pair<T>::value;

    template<typename T, typename = void>
    struct is_derived_from_pair : std::false_type
    {
    };

    // Specialization for std::pair
    template<typename T>
    struct is_derived_from_pair<T, std::void_t<typename T::first_type, typename T::second_type>> : std::true_type
    {
    };

    // Helper variable template for is_derived_from_pair
    template<typename T>
    constexpr bool is_derived_from_pair_v = is_derived_from_pair<T>::value;

} // namespace redGrapes::traits
