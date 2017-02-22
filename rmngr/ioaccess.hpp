
#pragma once

#include <rmngr/dependency_manager.hpp>

namespace rmngr
{

struct IOAccess
{
    enum Id
    {
        root,
        read,
        write,
        aadd,
        amul,
    };

    template <typename DependencyManager>
    static void addOperation(DependencyManager& m, Id const& id, bool atomic)
    {
        m.add_vertex(id);
        m.add_dependency(root, id);
        m.add_dependency(id, root);
        if(! atomic)
            m.add_dependency(id, id);
    }

    template <typename DependencyManager>
    static void build_dependencies(DependencyManager& m)
    {
        m.add_vertex(root);
        addOperation(m, read, true);
        addOperation(m, write, false);
        addOperation(m, aadd, true);
        addOperation(m, amul, true);
    }
}; // struct IOAccess

} // namespace rmngr

