#include "optical_flow.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <algorithm>
#include <vector>

namespace igl
{
unsigned char double_2_unsignedchar(const double d)
{
    return round(std::max(std::min(1., d), 0.) * 255);
}

double unsignedchar_2_double(const unsigned char c)
{
    return (double)c / 255.0;
}

double masked_position(const OPTICALData &o, const int row, const int col)
{
    double x_flow = std::max(std::min(o.u(row, col), o.height - row * 1.0 - 1), -row * 1.0);
    double y_flow = std::max(std::min(o.v(row, col), o.width - col * 1.0 - 1), -col * 1.0);
    return o.image2(round(row + round(x_flow)), round(col + round(y_flow)));
}

IGL_INLINE void median_filter(Eigen::MatrixXd &m)
{
    Eigen::MatrixXd filtered_m = m;
    std::vector<double> neighbors;
    neighbors.resize(9);
    for (int w = 1; w < m.cols() - 1; w++)
    {
        for (int h = 1; h < m.rows() - 1; h++)
        {
            neighbors[0] = m(h - 1, w - 1);
            neighbors[1] = m(h - 1, w);
            neighbors[2] = m(h - 1, w + 1);
            neighbors[3] = m(h, w - 1);
            neighbors[4] = m(h, w);
            neighbors[5] = m(h, w + 1);
            neighbors[6] = m(h + 1, w - 1);
            neighbors[7] = m(h + 1, w);
            neighbors[8] = m(h + 1, w + 1);
            std::sort(neighbors.begin(), neighbors.end());
            filtered_m(h, w) = neighbors[4];
        }
    }
    neighbors.resize(6);
    // dealing with the borders
    for (int w = 1; w < m.cols() - 1; w++)
    {
        neighbors[0] = m(0, w - 1);
        neighbors[1] = m(0, w);
        neighbors[2] = m(0, w + 1);
        neighbors[3] = m(1, w - 1);
        neighbors[4] = m(1, w);
        neighbors[5] = m(1, w + 1);
        std::sort(neighbors.begin(), neighbors.end());
        filtered_m(0, w) = (neighbors[2] + neighbors[3]) / 2;
    }
    for (int w = 1; w < m.cols() - 1; w++)
    {
        neighbors[0] = m(m.rows() - 1, w - 1);
        neighbors[1] = m(m.rows() - 1, w);
        neighbors[2] = m(m.rows() - 1, w + 1);
        neighbors[3] = m(m.rows() - 2, w - 1);
        neighbors[4] = m(m.rows() - 2, w);
        neighbors[5] = m(m.rows() - 2, w + 1);
        std::sort(neighbors.begin(), neighbors.end());
        filtered_m(m.rows() - 1, w) = (neighbors[2] + neighbors[3]) / 2;
    }
    for (int h = 1; h < m.rows() - 1; h++)
    {
        neighbors[0] = m(h - 1, 0);
        neighbors[1] = m(h, 0);
        neighbors[2] = m(h + 1, 0);
        neighbors[3] = m(h - 1, 1);
        neighbors[4] = m(h, 1);
        neighbors[5] = m(h + 1, 1);
        std::sort(neighbors.begin(), neighbors.end());
        filtered_m(h, 0) = (neighbors[2] + neighbors[3]) / 2;
    }
    for (int h = 1; h < m.rows() - 1; h++)
    {
        neighbors[0] = m(h - 1, m.cols() - 1);
        neighbors[1] = m(h, m.cols() - 1);
        neighbors[2] = m(h + 1, m.cols() - 1);
        neighbors[3] = m(h - 1, m.cols() - 2);
        neighbors[4] = m(h, m.cols() - 2);
        neighbors[5] = m(h + 1, m.cols() - 2);
        std::sort(neighbors.begin(), neighbors.end());
        filtered_m(h, m.cols() - 1) = (neighbors[2] + neighbors[3]) / 2;
    }
    neighbors.resize(4);
    neighbors[0] = m(0, 0);
    neighbors[1] = m(0, 1);
    neighbors[2] = m(1, 0);
    neighbors[3] = m(1, 1);
    std::sort(neighbors.begin(), neighbors.end());
    filtered_m(0, 0) = (neighbors[1] + neighbors[2]) / 2;
    neighbors[0] = m(0, m.cols() - 1);
    neighbors[1] = m(0, m.cols() - 2);
    neighbors[2] = m(1, m.cols() - 1);
    neighbors[3] = m(1, m.cols() - 2);
    std::sort(neighbors.begin(), neighbors.end());
    filtered_m(0, m.cols() - 1) = (neighbors[1] + neighbors[2]) / 2;
    neighbors[0] = m(m.rows() - 1, 0);
    neighbors[1] = m(m.rows() - 1, 1);
    neighbors[2] = m(m.rows() - 2, 0);
    neighbors[3] = m(m.rows() - 2, 1);
    std::sort(neighbors.begin(), neighbors.end());
    filtered_m(m.rows() - 1, 0) = (neighbors[1] + neighbors[2]) / 2;
    neighbors[0] = m(0, 0);
    neighbors[1] = m(0, 1);
    neighbors[2] = m(1, 0);
    neighbors[3] = m(1, 1);
    std::sort(neighbors.begin(), neighbors.end());
    filtered_m(0, 0) = (neighbors[1] + neighbors[2]) / 2;
    neighbors[0] = m(m.rows() - 1, m.cols() - 1);
    neighbors[1] = m(m.rows() - 1, m.cols() - 2);
    neighbors[2] = m(m.rows() - 2, m.cols() - 1);
    neighbors[3] = m(m.rows() - 2, m.cols() - 2);
    std::sort(neighbors.begin(), neighbors.end());
    filtered_m(m.rows() - 1, m.cols() - 1) = (neighbors[1] + neighbors[2]) / 2;
    m = filtered_m;
}

IGL_INLINE void write_image(Eigen::MatrixXd im, std::string file_name)
{
    const int w = im.cols();                          // Image width
    const int h = im.rows();                          // Image height
    const int comp = 1;                               // 3 Channels Red, Green, Blue, Alpha
    const int stride_in_bytes = w * comp;             // Lenght of one row in bytes
    std::vector<unsigned char> data(w * h * comp, 0); // The image itself;

    for (unsigned wi = 0; wi < w; ++wi)
    {
        for (unsigned hi = 0; hi < h; ++hi)
        {
            data[(hi * w) + wi] = double_2_unsignedchar(im(hi, wi));
        }
    }
    stbi_write_png(file_name.c_str(), w, h, comp, data.data(), stride_in_bytes);
}

IGL_INLINE void load_image1(OPTICALData &o, std::string file_name)
{
    int n;
    unsigned char *data = stbi_load(file_name.c_str(), &o.width, &o.height, &n, 0);
    o.image1.resize(o.height, o.width);
    for (unsigned wi = 0; wi < o.width; ++wi)
    {
        for (unsigned hi = 0; hi < o.height; ++hi)
        {
            o.image1(hi, wi) = unsignedchar_2_double(data[hi * o.width + wi]);
        }
    }

    o.Ix.resize(o.height, o.width);
    o.Iy.resize(o.height, o.width);
    o.It.resize(o.height, o.width);
    o.ubar.resize(o.height, o.width);
    o.vbar.resize(o.height, o.width);
    o.u.resize(o.height, o.width);
    o.v.resize(o.height, o.width);
    o.lhs.resize(o.height * o.width * 2, o.height * o.width * 2);
    for (int i = 0; i < o.height; i++)
    {
        for (int j = 0; j < o.width; j++)
        {
            o.Ix(i, j) = 0.0;
            o.Iy(i, j) = 0.0;
            o.It(i, j) = 0.0;
            o.ubar(i, j) = 0.0;
            o.vbar(i, j) = 0.0;
            o.u(i, j) = 0.0;
            o.v(i, j) = 0.0;
        }
    }
}

IGL_INLINE void load_image2(OPTICALData &o, std::string file_name)
{
    int n;
    unsigned char *data = stbi_load(file_name.c_str(), &o.width, &o.height, &n, 0);
    o.image2.resize(o.height, o.width);
    for (unsigned wi = 0; wi < o.width; ++wi)
    {
        for (unsigned hi = 0; hi < o.height; ++hi)
        {
            o.image2(hi, wi) = unsignedchar_2_double(data[hi * o.width + wi]);
        }
    }
}

IGL_INLINE void compute_ix(OPTICALData &o)
{
    for (int w = 0; w < o.width - 1; w++)
    {
        for (int h = 0; h < o.height - 1; h++)
        {
            o.Ix(h, w) = (o.image1(h, w + 1) - o.image1(h, w) + o.image1(h + 1, w + 1) - o.image1(h + 1, w) + masked_position(o, h, w + 1) - masked_position(o, h, w) + masked_position(o, h + 1, w + 1) - masked_position(o, h + 1, w)) / 4;
        }
    }
    // set the boundary now
    for (int h = 0; h < o.height - 1; h++)
    {
        o.Ix(h, o.width - 1) = o.Ix(h, o.width - 2);
    }

    for (int w = 0; w < o.width - 1; w++)
    {
        o.Ix(o.height - 1, w) = (o.image1(o.height - 1, w + 1) - o.image1(o.height - 1, w) + masked_position(o, o.height - 1, w + 1) - masked_position(o, o.height - 1, w)) / 4;
    }
    // now do the corner
    o.Ix(o.height - 1, o.width - 1) = o.Ix(o.height - 1, o.width - 2);
}

IGL_INLINE void compute_iy(OPTICALData &o)
{
    for (int w = 0; w < o.width - 1; w++)
    {
        for (int h = 0; h < o.height - 1; h++)
        {
            o.Iy(h, w) = (o.image1(h + 1, w) - o.image1(h, w) + o.image1(h + 1, w + 1) - o.image1(h, w + 1) + masked_position(o, h + 1, w) - masked_position(o, h, w) + masked_position(o, h + 1, w + 1) - masked_position(o, h, w + 1)) / 4;
        }
    }
    // set the boundary now
    for (int h = 0; h < o.height - 1; h++)
    {
        o.Iy(h, o.width - 1) = (o.image1(h + 1, o.width - 1) - o.image1(h, o.width - 1) + masked_position(o, h + 1, o.width - 1) - masked_position(o, h, o.width - 1)) / 4;
    }

    for (int w = 0; w < o.width - 1; w++)
    {
        o.Iy(o.height - 1, w) = o.Iy((w + 1) * o.height - 2);
    }
    // now do the corner
    o.Iy(o.height - 1, o.width - 1) = o.Iy(o.height - 2, o.width - 1);
}

IGL_INLINE void compute_it(OPTICALData &o)
{
    for (int w = 0; w < o.width - 1; w++)
    {
        for (int h = 0; h < o.height - 1; h++)
        {
            o.It(h, w) = (masked_position(o, h, w) - o.image1(h, w) + masked_position(o, h + 1, w) - o.image1(h + 1, w) + masked_position(o, h, w + 1) - o.image1(h, w + 1) + masked_position(o, h + 1, w + 1) - o.image1(h + 1, w + 1)) / 4;
        }
    }

    for (int h = 0; h < o.height - 1; h++)
    {
        o.It(h, o.width - 1) = (masked_position(o, h, o.width - 1) - o.image1(h, o.width - 1) + masked_position(o, h + 1, o.width - 1) - o.image1(h + 1, o.width - 1)) / 4;
    }
    for (int w = 0; w < o.width - 1; w++)
    {
        o.It(o.height - 1, w) = (masked_position(o, o.height - 1, w) - o.image1(o.height - 1, w) + masked_position(o, o.height - 1, w + 1) - o.image1(o.height - 1, w + 1)) / 4;
    }
    o.It(o.height - 1, o.width - 1) = masked_position(o, o.height - 1, o.width - 1) - o.image1(o.height - 1, o.width - 1);
}

IGL_INLINE void compute_ubar(OPTICALData &o)
{
    for (int w = 1; w < o.width - 1; w++)
    {
        for (int h = 1; h < o.height - 1; h++)
        {
            o.ubar(h, w) = (o.u(h + 1, w) + o.u(h, w + 1) + o.u(h - 1, w) + o.u(h, w - 1)) / 6 + (o.u(h - 1, w - 1) + o.u(h - 1, w + 1) + o.u(h + 1, w - 1) + o.u(h + 1, w + 1)) / 12;
        }
    }

    // get the four borders
    for (int h = 1; h < o.height - 1; h++)
    {
        o.ubar(h, 0) = (o.u(h - 1, 0) + o.u(h + 1, 0) + o.u(h, 1)) / 6 + (o.u(h - 1, 1) + o.u(h + 1, 1)) / 12;
    }
    for (int h = 1; h < o.height - 1; h++)
    {
        o.ubar(h, o.width - 1) = (o.u(h - 1, o.width - 1) + o.u(h + 1, o.width - 1) + o.u(h, o.width - 2)) / 6 + (o.u(h - 1, o.width - 2) + o.u(h + 1, o.width - 2)) / 12;
    }
    for (int w = 1; w < o.width - 1; w++)
    {
        o.ubar(0, w) = (o.u(0, w - 1) + o.u(0, w + 1) + o.u(1, w)) / 6 + (o.u(1, w - 1) + o.u(1, w + 1)) / 12;
    }
    for (int w = 1; w < o.width - 1; w++)
    {
        o.ubar(o.height - 1, w) = (o.u(o.height - 1, w - 1) + o.u(o.height - 1, w + 1) + o.u(o.height - 2, w)) / 6 + (o.u(o.height - 2, w - 1) + o.u(o.height - 2, w + 1)) / 12;
    }

    // get the four corners
    o.ubar(0, 0) = (o.u(1, 0) + o.u(0, 1)) / 6 + o.u(1, 1) / 12;
    o.ubar(o.height - 1, 0) = (o.u(o.height - 2, 0) + o.u(o.height - 1, 1)) / 6 + o.u(o.height - 2, 1) / 12;
    o.ubar(0, o.width - 1) = (o.u(0, o.width - 2) + o.u(1, o.height - 1)) / 6 + o.u(1, o.width - 2) / 12;
    o.ubar(o.height - 1, o.width - 1) = (o.u(o.height - 2, o.width - 1) + o.u(o.height - 1, o.width - 2)) / 6 + o.u(o.height - 2, o.width - 2) / 12;
}

IGL_INLINE void compute_vbar(OPTICALData &o)
{
    for (int w = 1; w < o.width - 1; w++)
    {
        for (int h = 1; h < o.height - 1; h++)
        {
            o.vbar(h, w) = (o.v(h + 1, w) + o.v(h, w + 1) + o.v(h - 1, w) + o.v(h, w - 1)) / 6 + (o.v(h - 1, w - 1) + o.v(h - 1, w + 1) + o.v(h + 1, w - 1) + o.v(h + 1, w + 1)) / 12;
        }
    }

    // get the four borders
    for (int h = 1; h < o.height - 1; h++)
    {
        o.vbar(h, 0) = (o.v(h - 1, 0) + o.v(h + 1, 0) + o.v(h, 1)) / 6 + (o.v(h - 1, 1) + o.v(h + 1, 1)) / 12;
    }
    for (int h = 1; h < o.height - 1; h++)
    {
        o.vbar(h, o.width - 1) = (o.v(h - 1, o.width - 1) + o.v(h + 1, o.width - 1) + o.v(h, o.width - 2)) / 6 + (o.v(h - 1, o.width - 2) + o.v(h + 1, o.width - 2)) / 12;
    }
    for (int w = 1; w < o.width - 1; w++)
    {
        o.vbar(0, w) = (o.v(0, w - 1) + o.v(0, w + 1) + o.v(1, w)) / 6 + (o.v(1, w - 1) + o.v(1, w + 1)) / 12;
    }
    for (int w = 1; w < o.width - 1; w++)
    {
        o.vbar(o.height - 1, w) = (o.v(o.height - 1, w - 1) + o.v(o.height - 1, w + 1) + o.v(o.height - 2, w)) / 6 + (o.v(o.height - 2, w - 1) + o.v(o.height - 2, w + 1)) / 12;
    }

    // get the four corners
    o.vbar(0, 0) = (o.v(1, 0) + o.v(0, 1)) / 6 + o.v(1, 1) / 12;
    o.vbar(o.height - 1, 0) = (o.v(o.height - 2, 0) + o.v(o.height - 1, 1)) / 6 + o.v(o.height - 2, 1) / 12;
    o.vbar(0, o.width - 1) = (o.v(0, o.width - 2) + o.v(1, o.height - 1)) / 6 + o.v(1, o.width - 2) / 12;
    o.vbar(o.height - 1, o.width - 1) = (o.v(o.height - 2, o.width - 1) + o.v(o.height - 1, o.width - 2)) / 6 + o.v(o.height - 2, o.width - 2) / 12;
}

IGL_INLINE void build_lhs(OPTICALData &o)
{
    compute_ix(o);
    compute_iy(o);
    compute_it(o);
    std::cout << "finished computing\n";
    double a2 = std::pow(o.alpha, 2);
    std::vector<Eigen::Triplet<double>> trip;
    trip.reserve(o.height * o.width * 3);
    Eigen::MatrixXd x_squared = o.Ix.cwiseProduct(o.Ix) + Eigen::MatrixXd::Constant(o.height, o.width, a2);
    Eigen::MatrixXd y_squared = o.Iy.cwiseProduct(o.Iy) + Eigen::MatrixXd::Constant(o.height, o.width, a2);
    Eigen::MatrixXd xy = o.Ix.cwiseProduct(o.Iy);
    int index = 0;
    for (int w = 0; w < o.width; w++)
    {
        for (int h = 0; h < o.height; h++)
        {
            trip.push_back(Eigen::Triplet<double>(index, index, x_squared(h, w)));
            index++;
        }
    }
    int index2 = index;
    for (int w = 0; w < o.width; w++)
    {
        for (int h = 0; h < o.height; h++)
        {
            trip.push_back(Eigen::Triplet<double>(index, index, y_squared(h, w)));
            index++;
        }
    }
    for (int w = 0; w < o.width; w++)
    {
        for (int h = 0; h < o.height; h++)
        {
            trip.push_back(Eigen::Triplet<double>(index2, index2 - o.height * o.width, xy(h, w)));
            index2++;
        }
    }
    std::cout << "finished triplet\n";
    o.lhs.setFromTriplets(trip.begin(), trip.end());
    o.lhs.makeCompressed();
}

IGL_INLINE void build_rhs(OPTICALData &o)
{
    if (o.rhs.size() == 0)
    {
        o.rhs.resize(o.height * o.width * 2);
    }
    compute_ubar(o);
    compute_vbar(o);
    Eigen::MatrixXd rhs1 = std::pow(o.alpha, 2) * o.ubar - o.Ix.cwiseProduct(o.It);
    Eigen::MatrixXd rhs2 = std::pow(o.alpha, 2) * o.vbar - o.Iy.cwiseProduct(o.It);

    int index = 0;
    for (int w = 0; w < o.width; w++)
    {
        for (int h = 0; h < o.height; h++)
        {
            o.rhs(index) = rhs1(h, w);
            index++;
        }
    }
    for (int w = 0; w < o.width; w++)
    {
        for (int h = 0; h < o.height; h++)
        {
            o.rhs(index) = rhs2(h, w);
            index++;
        }
    }
}

IGL_INLINE void solve_flow(OPTICALData &o)
{
    std::cout << "Building left hand side\n";
    build_lhs(o);
    std::cout << "Building right hand side\n";
    build_rhs(o);

    std::vector<double> solved(o.height * o.width * 2); // because we solve for u and v together
    if (o.first_called)
    {
        std::cout << "First called, supporting matrix and doing symbolic factorization\n";
        pardiso_init(o.pardiso_data);
        std::cout << "finished init\n";
        pardiso_support_matrix(o.pardiso_data, o.lhs);
        std::cout << "finished supporting matrix\n";
        pardiso_symbolic_factor(o.pardiso_data);
        std::cout << "finished symbolic factor\n";
    }
    else
    {
        std::cout << "Supporting data\n";
        pardiso_support_value(o.pardiso_data, o.lhs.valuePtr());
    }
    std::cout << "Doing numeric factorization\n";
    pardiso_numeric_factor(o.pardiso_data);
    std::cout << "Doing solve\n";
    pardiso_solve(o.pardiso_data, solved.data(), o.rhs.data());

    std::cout << "Putting back to matrix\n";
    o.u = Eigen::Map<Eigen::MatrixXd>(solved.data(), o.height, o.width);
    o.v = Eigen::Map<Eigen::MatrixXd>(solved.data() + o.height * o.width, o.height, o.width);
    o.first_called = false;

    // clip the values
    // for (int w = 0; w < o.width; w++)
    // {
    //     for (int h = 0; h < o.height; h++)
    //     {
    //         o.u(h, w) = round(std::min(std::max(o.u(h, w), -w * 1.0), o.width - w * 1.0 - 1.0));
    //         o.v(h, w) = round(std::min(std::max(o.v(h, w), -h * 1.0), o.height - h * 1.0 - 1.0));
    //     }
    // }
    // median_filter(o.u);
    // median_filter(o.v);
}

} // namespace igl