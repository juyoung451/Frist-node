#include <chrono>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"

using namespace std::chrono_literals;

class NumberPublisher : public rclcpp::Node
{
public:
    NumberPublisher() : Node("number_publisher"), count_(0)
    {
        // "number_topic" 토픽으로 Int32 타입 퍼블리셔 생성, 큐 사이즈 10
        publisher_ = this->create_publisher<std_msgs::msg::Int32>("number_topic", 10);

        // 500ms 마다 timer_callback 호출
        timer_ = this->create_wall_timer(
            500ms, [this]() { this->timer_callback(); });
    }

private:
    void timer_callback()
    {
        auto message = std_msgs::msg::Int32();
        message.data = count_++;  // count_ 값을 메시지로 전송 후 증가

        RCLCPP_INFO(this->get_logger(), "Publishing: %d", message.data);

        publisher_->publish(message);
    }

    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    int count_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<NumberPublisher>());
    rclcpp::shutdown();
    return 0;
}
