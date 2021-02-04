

#pragma once

#include <typeinfo>

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
        spdlog::warn("trait `redGrapes::BuildProperties` is not implemented for {}", typeid(T).name());
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

} // namespace trait

} // namespace redGrapes

