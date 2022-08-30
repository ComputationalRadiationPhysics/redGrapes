/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/**
 * @file redGrapes/access/combine.hpp
 */

#pragma once

#include <array>
#include <utility>
#include <iostream>

#include <fmt/format.h>

namespace redGrapes
{
namespace access
{

// TODO: better tag names
struct And_t {};
struct Or_t {};

template <
    typename Access,
    size_t N,
    typename Op = Or_t
>
struct ArrayAccess : std::array<Access, N>
{
    ArrayAccess()
    {
        for( Access & a : *this )
            a = Access();
    }

    ArrayAccess(std::array<Access, N> const & a)
      : std::array<Access, N>(a) {}

    bool is_synchronizing() const
    {
        for(std::size_t i = 0; i < N; ++i)
            if( !(*this)[i].is_synchronizing() )
                return false;

        return true;
    }
    
    //! both array accesses are only serial if all element pairs are serial
    static bool
    is_serial(
        ArrayAccess<Access, N, Or_t> const & a,
        ArrayAccess<Access, N, Or_t> const & b
    )
    {
        for(std::size_t i = 0; i < N; ++i)
            if(Access::is_serial(a[i], b[i]))
                return true;

        return false;
    }

    //! both array accesses are serial if at least one element pair is serial
    static bool
    is_serial(
        ArrayAccess<Access, N, And_t> const & a,
        ArrayAccess<Access, N, And_t> const & b
    )
    {
        for(std::size_t i = 0; i < N; ++i)
            if(! Access::is_serial(a[i], b[i]))
                return false;

        return true;
    }

    bool
    is_superset_of( ArrayAccess<Access, N, Op> const & a ) const
    {
        for(std::size_t i = 0; i < N; ++i)
            if( ! (*this)[i].is_superset_of( a[i] ) )
                return false;

        return true;
    }

    bool
    operator==( ArrayAccess<Access, N, Op> const & other ) const
    {
        for(std::size_t i = 0; i < N; ++i)
            if( ! ( (*this)[i] == other[i] ) )
                return false;

        return true;
    }
}; // struct ArrayAccess

template <
    typename Acc1,
    typename Acc2,
    typename Op = And_t
>
struct CombineAccess : std::pair<Acc1, Acc2>
{
    CombineAccess()
        : std::pair<Acc1, Acc2>( Acc1(), Acc2() ) {}

    CombineAccess(Acc1 a)
        : std::pair<Acc1, Acc2>(a, Acc2()) {}

    CombineAccess(Acc1 a, Acc2 b)
        : std::pair<Acc1, Acc2>(a,b) {}

    static bool
    is_serial(
        CombineAccess<Acc1, Acc2, And_t> const & a,
        CombineAccess<Acc1, Acc2, And_t> const & b
    )
    {
        return (
            Acc1::is_serial(a.first, b.first) &&
            Acc2::is_serial(a.second, b.second)
        );
    }

    bool is_synchronizing() const
    {
        return this->first.is_synchronizing() && this->second.is_synchronizing();
    }
    
    static bool
    is_serial(
        CombineAccess<Acc1, Acc2, Or_t> const & a,
        CombineAccess<Acc1, Acc2, Or_t> const & b
    )
    {
        return (
            Acc1::is_serial(a.first, b.first) ||
            Acc2::is_serial(a.second, b.second)
        );
    }

    bool
    is_superset_of( CombineAccess<Acc1, Acc2, Op> const & a ) const
    {
        return (
            this->first.is_superset_of(a.first) &&
            this->second.is_superset_of(a.second)
        );
    }

    bool
    operator==( CombineAccess<Acc1, Acc2, Op> const & other ) const
    {
        return (
             this->first == other.first &&
             this->second == other.second
        );
    }
}; // struct CombineAccess

} // namespace access

} // namespace redGrapes


template <
    typename Access,
    size_t N
>
struct fmt::formatter<
    redGrapes::access::ArrayAccess< Access, N, redGrapes::access::And_t >
>
{
    constexpr auto parse( format_parse_context& ctx )
    {
        return ctx.begin();
    }

    template < typename FormatContext >
    auto format(
        redGrapes::access::ArrayAccess< Access, N, redGrapes::access::And_t > const & acc,
        FormatContext & ctx
    )
    {
        auto out = ctx.out();
        out = fmt::format_to( out, "{{ \"and\" : [" );

        for( auto it = acc.begin(); it != acc.end(); )
        {
            out = fmt::format_to( out, "{}", *it );
            if( ++it != acc.end() )
                out = fmt::format_to( out, ", " );
        }

        out = fmt::format_to( out, "] }}" );
        return out;
    }
};

template <
    typename Access,
    size_t N
>
struct fmt::formatter<
    redGrapes::access::ArrayAccess< Access, N, redGrapes::access::Or_t >
>
{
    constexpr auto parse( format_parse_context& ctx )
    {
        return ctx.begin();
    }

    template < typename FormatContext >
    auto format(
        redGrapes::access::ArrayAccess< Access, N, redGrapes::access::Or_t > const & acc,
        FormatContext & ctx
    )
    {
        auto out = ctx.out();
        out = fmt::format_to( out, "{{ \"or\" : [" );

        for( auto it = acc.begin(); it != acc.end(); )
        {
            out = fmt::format_to( out, "{}", *it );
            if( ++it != acc.end() )
                out = fmt::format_to( out, ", " );
        }

        out = fmt::format_to( out, "] }}" );
        return out;
    }
};

template <
    typename Acc1,
    typename Acc2
>
struct fmt::formatter<
    redGrapes::access::CombineAccess< Acc1, Acc2, redGrapes::access::And_t >
>
{
    constexpr auto parse( format_parse_context& ctx )
    {
        return ctx.begin();
    }

    template < typename FormatContext >
    auto format(
        redGrapes::access::CombineAccess< Acc1, Acc2, redGrapes::access::And_t > const & acc,
        FormatContext & ctx
    )
    {
        return fmt::format_to( ctx.out(), "{{ \"and\" : [ {}, {} ] }}", acc.first, acc.second );
    }
};

template <
    typename Acc1,
    typename Acc2
>
struct fmt::formatter<
    redGrapes::access::CombineAccess< Acc1, Acc2, redGrapes::access::Or_t >
>
{
    constexpr auto parse( format_parse_context& ctx )
    {
        return ctx.begin();
    }

    template < typename FormatContext >
    auto format(
        redGrapes::access::CombineAccess< Acc1, Acc2, redGrapes::access::Or_t > const & acc,
        FormatContext & ctx
    )
    {
        return fmt::format_to( ctx.out(), "{{ \"or\" : [ {}, {} ] }}", acc.first, acc.second );
    }
};


