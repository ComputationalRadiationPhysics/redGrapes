
/**
 * @file rmngr/property/label.hpp
 */

#pragma once

namespace rmngr
{

struct LabelProperty
{
    std::string label;

    template < typename PropertiesBuilder >
    struct Builder
    {
        PropertiesBuilder & builder;

        Builder( PropertiesBuilder & b )
            : builder(b)
        {}

        PropertiesBuilder label( std::string const & l )
        {
            builder.prop.label = l;
            return builder;
        }
    };

    struct Patch {};
    void apply_patch( Patch const & ) {};
};

} // namespace rmngr

