
/**
 * @file rmngr/graph_util.hpp
 */

#pragma once

#include <utility> // std::pair
#include <boost/graph/graph_traits.hpp>

namespace rmngr
{

/**
 * get vertex-property from vertex-descriptor
 */
template <typename Graph>
typename Graph::vertex_property_type
graph_get(
    typename boost::graph_traits<Graph>::vertex_descriptor v,
    Graph & graph
)
{
    return boost::get(boost::vertex_bundle, graph)[v];
}

/**
 * find vertex-descriptor from property
 *
 * TODO: overload for boost::labeled_graph
 *
 * @return pair of (vertex-descriptor, true) if vertex with
 *         property exists, else (_, false)
 */
template <typename Graph>
std::pair<
    typename boost::graph_traits<Graph>::vertex_descriptor,
    bool
>
graph_find_vertex(
    typename Graph::vertex_property_type a,
    Graph & graph
)
{
    typename boost::graph_traits<Graph>::vertex_iterator it, end;

    for (boost::tie(it, end) = boost::vertices(graph); it != end; ++it)
    {
        if (graph_get(*it, graph) == a)
            return std::make_pair(*it, true);
    }

    return std::make_pair(*it, false);
}

} // namespace rmngr

