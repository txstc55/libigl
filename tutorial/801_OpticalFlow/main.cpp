#include <iostream>
#include <igl/optical_flow.h>
#include <Eigen/Dense>

int main()
{
    Eigen::MatrixXd im;
    im.resize(5, 5);
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            im(i, j) = 0;
        }
    }
    for (int i = 1; i < 4; i++)
    {
        for (int j = 1; j < 4; j++)
        {
            im(i - 1, j - 1) = (i * j * 1.0) / 9.0;
        }
    }

    igl::OPTICALData o;
    igl::write_image(im, "test_image1.png");

    Eigen::MatrixXd im2;
    im2.resize(5, 5);
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            im2(i, j) = 0;
        }
    }
    for (int i = 1; i < 4; i++)
    {
        for (int j = 1; j < 4; j++)
        {
            im2(i + 1, j + 1) = (i * j * 1.0) / 9.0;
        }
    }
    igl::write_image(im2, "test_image2.png");

    std::cout << "load image\n";
    igl::load_image1(o, "test_image1.png");
    igl::load_image2(o, "test_image2.png");

    std::cout << "Dimensions: " << o.height << " " << o.width << "\n";

    o.alpha = 1.0;
    for (int i = 0; i < 100; i++)
    {
        std::cout << "iteration " << i << "\n";
        igl::solve_flow(o);
    }
    std::cout << "Ix:\n"
              << o.Ix << "\n";
    std::cout << "Iy:\n"
              << o.Iy << "\n";

    std::cout << "Final u:\n"
              << o.u << "\n";
    std::cout << "Final v:\n"
              << o.v << "\n";

    o.outimage = o.image1;
    for (int w = 0; w < o.width; w++)
    {
        for (int h = 0; h < o.height; h++)
        {
            double x_flow = std::max(std::min(o.u(h, w), o.height - h * 1.0 - 1), -h * 1.0);
            double y_flow = std::max(std::min(o.v(h, w), o.width - w * 1.0 - 1), -w * 1.0);
            // std::cout<<h<<" "<<w<<" "<<x_flow<<" "<<round(x_flow)<<"\n";
            o.outimage(h, w) = o.image1(h + round(x_flow), w + round(y_flow));
        }
    }
    double v_max = o.v.maxCoeff();
    double v_min = o.v.minCoeff();
    double u_max = o.u.maxCoeff();
    double u_min = o.u.minCoeff();

    igl::write_image(o.outimage, "out.png");
    igl::write_image((o.u - Eigen::MatrixXd::Constant(o.height, o.width, u_min)) / (u_max - u_min), "u.png");
    igl::write_image((o.v - Eigen::MatrixXd::Constant(o.height, o.width, v_min)) / (v_max - v_min), "v.png");
    // std::cout << (o.u - o.v).norm() << " <------ the norm\n";
}