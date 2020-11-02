
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
        std::cout << "Warning: property builder not implemented for " << typeid(T).name() << std::endl;
    }
};

} // namespace trait

} // namespace redGrapes

