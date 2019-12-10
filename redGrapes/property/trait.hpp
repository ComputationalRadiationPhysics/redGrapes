

#pragma once

#include <redGrapes/resource/resource.hpp>

namespace redGrapes
{
namespace trait
{

template < typename T >
struct BuildProperties
{
    template <typename Builder>
    static void build(Builder & builder, T const & t)
    {
    }
};

} // namespace trait

} // namespace redGrapes

