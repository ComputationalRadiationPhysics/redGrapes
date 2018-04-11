
/**
 * @file rmngr/fieldresource.hpp
 */

#pragma once

#include <rmngr/access/field.hpp>
#include <rmngr/resource.hpp>

namespace rmngr
{

template <
    size_t dimension_t
>
struct FieldResource : Resource< access::FieldAccess<dimension_t> >
{
    using Resource<access::FieldAccess<dimension_t>>::make_access;
    ResourceAccess make_access(
        access::IOAccess io,
        std::array<access::AreaAccess, dimension_t> range
    )
    {
        return this->make_access(
                   access::FieldAccess<dimension_t>(
                       io,
                       access::ArrayAccess<
                           access::AreaAccess,
                           dimension_t
                       >(range)
                   )
               );
    }

#define OP(name)                                                                \
    inline ResourceAccess name (                                                \
        std::array<std::array<int,2>, dimension_t> const& range                 \
    )                                                                           \
    {                                                                           \
        return this->make_access(                                               \
                   access::IOAccess{access::IOAccess::name},                    \
                   *reinterpret_cast<                                           \
                       std::array<access::AreaAccess, dimension_t> const*       \
                    >(&range)                                                   \
               );                                                               \
    }

    OP(read)
    OP(write)
    OP(aadd)
    OP(amul)

#undef OP
}; // struct FieldResource

}; // namespace rmngr

