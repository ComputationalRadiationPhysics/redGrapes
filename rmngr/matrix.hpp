
#pragma once

#include <valarray>

namespace rmngr
{

template <typename T>
class Matrix : public std::valarray<T>
{
    public:
        Matrix()
            : width(0), height(0)
        {
        }

        Matrix(int width_, int height_)
        {
            this->resize(width_, height_);
        }

        void resize(int width_, int height_)
        {
            this->width = width_;
            this->height = height_;

            std::valarray<T>::resize(this->width * this->height);
        }

        T& operator()(int r, int c)
        {
            return (*this)[r * this->width + c];
        }

        void write_row(int r, std::valarray<T> const& data)
        {
            for(int c = 0; c < this->width; ++c)
                (*this)(r,c) = data[c];
        }

        void write_col(int c, std::valarray<T> const& data)
        {
            for(int r = 0; r < this->heigth; ++r)
                (*this)(r,c) = data[c];
        }

        std::valarray<T> row(int r) const
        {
            assert(r < this->height);
            return (*this)[ std::slice(r * this->width, this->width, 1) ];
        }

        std::valarray<T> col(int c) const
        {
            assert(c < this->width);
            return (*this)[ std::slice(c, this->height, this->width) ];
        }

    private:
        int width; // number of colums, length of a row
        int height; // number of rows, length of a column
}; // class Matrix

} // namespace rmngr

