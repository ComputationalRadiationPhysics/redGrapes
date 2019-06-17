
/**
 * @file rmngr/resource/ioresource.hpp
 */

#pragma once

#include <rmngr/access/io.hpp>
#include <rmngr/resource/resource.hpp>

namespace rmngr
{

struct IOResource : public Resource< access::IOAccess >
{
#define OP(name)                                                                \
    inline ResourceAccess name (void) const                                     \
    { return this->make_access(access::IOAccess{access::IOAccess::name}); }

    OP(read)
    OP(write)
    OP(aadd)
    OP(amul)

#undef OP
}; // struct IOResource

} // namespace rmngr

