

#pragma once

#include <typeinfo>
#include <boost/core/demangle.hpp>

namespace redGrapes
{
namespace trait
{

template <
    typename T,
    typename Sfinae = void
>
struct BuildProperties
{
    template <typename Builder>
    static void build(Builder & builder, T const & t)
    {
        spdlog::warn("trait `redGrapes::BuildProperties` is not implemented for {}", boost::core::demangle(typeid(T).name()));
    }
};

template <
    typename T
>
struct BuildProperties< std::reference_wrapper< T > >
{
    template <typename Builder>
    static void build(Builder & builder, std::reference_wrapper< T > const & t)
    {
        builder.add( t.get() );
    }
};

template <
    typename T
>
struct BuildProperties< T & >
{
    template <typename Builder>
    static void build(Builder & builder, T const & t)
    {
        builder.add( t );
    }
};

template <
    typename T
>
struct BuildProperties< T const & >
{
    template <typename Builder>
    static void build(Builder & builder, T const & t)
    {
        builder.add( t );
    }
};


// to avoid warnings
template <>
struct BuildProperties< int >
{
    template <typename Builder>
    static void build(Builder & builder, int const & t)
    {}
};

template <>
struct BuildProperties< unsigned int >
{
    template <typename Builder>
    static void build(Builder & builder, unsigned int const & t)
    {}
};

} // namespace trait

} // namespace redGrapes

