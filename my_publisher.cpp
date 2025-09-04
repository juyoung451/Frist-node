#include <chrono>                        // 시간 관련 (ms, s 같은 단위 사용)
#include <memory>                        // 스마트 포인터 (shared_ptr)
#include <string>                        // 문자열 (std::string)
#include "rclcpp/rclcpp.hpp"             // ROS2 C++ API 핵심
#include "std_msgs/msg/string.hpp"       // String 메시지 타입 정의

using namespace std::chrono_literals;   // 500ms, 1s 같은 리터럴 문법 사용

class MinimalPublisher : public rclcpp::Node
{
public:
    MinimalPublisher() : Node("minimal_publisher"), count_(0)
    {
        // "topic" 이라는 이름의 토픽으로 Publisher 생성 (큐 사이즈 10)
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);

        // 500ms마다 timer_callback 함수를 실행하는 타이머 생성
        timer_ = this->create_wall_timer(
            500ms, std::bind(&MinimalPublisher::timer_callback, this));
    }

private:
    void timer_callback()
    {
        // 메시지 객체 생성 (auto → std_msgs::msg::String)
        auto message = std_msgs::msg::String();
        message.data = "Hello, ROS2! " + std::to_string(count_++);

        // ROS2 로그(INFO 레벨) 출력
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());

        // Publisher를 통해 메시지 전송
        publisher_->publish(message);
    }

    // 멤버 변수들
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_; // 퍼블리셔 객체 (스마트 포인터)
    rclcpp::TimerBase::SharedPtr timer_;                            // 타이머 객체
    size_t count_;                                                  // 메시지 카운트
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);                                 // ROS2 초기화
    rclcpp::spin(std::make_shared<MinimalPublisher>());       // 노드를 실행 (콜백 처리 루프)
    rclcpp::shutdown();                                       // 종료 처리
    return 0;
}
