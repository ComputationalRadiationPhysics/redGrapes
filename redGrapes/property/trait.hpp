
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

} // namespace trait

} // namespace redGrapes

