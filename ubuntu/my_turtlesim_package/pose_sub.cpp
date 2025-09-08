#include "rclcpp/rclcpp.hpp"
#include "turtlesim/msg/pose.hpp"

class TurtlePose : public rclcpp::Node
{
    public:
        TurtlePose() : Node("turtle_pose_sub")
        {
            subscriber_ = this->create_subscription<turtlesim::msg::Pose>("/turtle1/pose", 10, std::bind(&TurtlePose::subscriber_callback, this, std::placeholders::_1));
        }

    private:
        void subscriber_callback(const turtlesim::msg::Pose::SharedPtr msg) const
        {
            printf("Turle Pose : x: %.2f, y: %.2f \n", msg->x, msg->y);
        }

        rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr subscriber_;
    
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TurtlePose>());
    rclcpp::shutdown();

    return 0;
}