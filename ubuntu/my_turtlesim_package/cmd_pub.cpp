#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"

using namespace std::chrono_literals;

class TurtleCmdVelPublisher : public rclcpp::Node
{
    public:
        TurtleCmdVelPublisher(): Node("turtle_cmd_vel_publisher")
        {
            publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("/turtle1/cmd_vel", 10);
            timer_ = this->create_wall_timer(500ms,std::bind(&TurtleCmdVelPublisher::publish_cmd_vel, this));

        }
    private:
        void publish_cmd_vel()
        {
            auto twist_msg = geometry_msgs::msg::Twist();
            twist_msg.linear.x = 0.5;

            publisher_->publish(twist_msg);

        }

        rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr publisher_;
        rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TurtleCmdVelPublisher>());
    rclcpp::shutdown();

    return 0;
}