
#pragma once

#include <list>
#include <unordered_map>
#include <boost/graph/subgraph.hpp>

namespace rmngr
{

struct AllSequential
{
    template <typename ID>
    bool is_sequential(ID, ID)
    {
        return true;
    }
};

struct AllParallel
{
    template <typename ID>
    bool is_sequential(ID, ID)
    {
        return false;
    }
};

} // namespace rmngr

