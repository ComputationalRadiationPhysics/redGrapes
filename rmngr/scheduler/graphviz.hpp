
#pragma once

#include <rmngr/scheduler/scheduler.hpp>

namespace rmngr
{

/**
 * Prints the Scheduling-Graph in a new file.
 * ColorProperty must be used in a SchedulingPolicy
 * and provide std::string color()
 */
template < typename ColorProperty >
struct GraphvizWriter : DefaultSchedulingPolicy
{
    struct ProtoProperty
    {
        std::string label;
    };

    template <typename Graph>
    void update( Graph & graph, SchedulerInterface & scheduler )
    {
        static int step = 0;
        ++step;
        std::string name = std::string( "Step " ) + std::to_string( step );
        std::string path = std::string( "step_" ) + std::to_string( step ) +
                           std::string( ".dot" );
        std::cout << "write schedulinggraph to " << path << std::endl;
        std::ofstream file( path );

        graph.write_graphviz(
            file,
            boost::make_function_property_map< ProtoProperty >(
                []( ProtoProperty const & s ) { return s.label; }
            ),
            boost::make_function_property_map< ColorProperty >(
                []( ColorProperty const & s ) { return s.color(); }
            ),
            name
        );

        file.close();
    }
};

}; // namespace rmngr

