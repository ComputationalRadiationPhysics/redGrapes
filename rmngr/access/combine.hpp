
/**
 * @file rmngr/access/combine.hpp
 */

#pragma once

#include <array>
#include <utility>
#include <iostream>

namespace rmngr
{
namespace access
{

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

    friend std::ostream& operator<<(std::ostream& out, ArrayAccess<Access, N, Op> const& a)
    {
        out << "ArrayAccess::{" << std::endl;
	for(std::size_t i = 0; i < N; ++i)
	    out << a[i] << "," << std::endl;
	out << "}";
	return out;
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
      : std::pair<Acc1, Acc2>( Acc1(), Acc2() )
    {}

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

    friend std::ostream& operator<<(std::ostream& out, CombineAccess<Acc1, Acc2, Op> const& a)
    {
        out << "CombineAccess::{" << std::endl;
	out << a.first << ";" << std::endl;
	out << a.second << ";" << std::endl;
	out << "}";
	return out;
    }
}; // struct CombineAccess

} // namespace access

} // namespace rmngr

