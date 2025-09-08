#include "rclcpp/rclcpp.hpp"
#include "turtlesim/msg/pose.hpp"
#include "geometry_msgs/msg/twist.hpp"

using namespace std::chrono_literals;

class TurtlePose : public rclcpp::Node
{
    public:
        TurtlePose() : Node("turtle_pose_sub")
        {
            subscriber_ = this->create_subscription<turtlesim::msg::Pose>("/turtle1/pose", 10, std::bind(&TurtlePose::subscriber_callback, this, std::placeholders::_1));
            publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/turtle1/cmd_vel", 10);
        }

    private:
        void subscriber_callback(const turtlesim::msg::Pose::SharedPtr msg) const
        {
            auto twist_msg = geometry_msgs::msg::Twist();
            if(msg->x != 11.0)
            {
                printf("Turle Pose : x: %.1f, y: %.1f \n", msg->x, msg->y);
                twist_msg.linear.x = 11.0 - msg->x;
                publisher_->publish(twist_msg);
            }
            else
            {
                twist_msg.linear.x = 0;
                publisher_->publish(twist_msg);
                printf("Stop\n");
            }
        }

        rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr subscriber_;
        rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
        rclcpp::TimerBase::SharedPtr timer_;

    
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TurtlePose>());
    rclcpp::shutdown();

    return 0;
}