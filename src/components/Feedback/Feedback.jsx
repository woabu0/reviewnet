import React from "react";
import feedback from "./feedbackData.json";
import { Swiper, SwiperSlide } from "swiper/react";
import { Autoplay, Navigation } from "swiper/modules";
import "swiper/css";
import "swiper/css/navigation";
import { motion } from "framer-motion";

export const Feedback = () => {
  return (
    <div id="feedback">
      <motion.div
        initial={{ y: -50, opacity: 0 }}
        whileInView={{ y: 0, opacity: 1 }}
        transition={{ ease: "linear", duration: 0.5 }}
      >
        <h1 className="mt-[50px] lg:mt-[74px] text-[24px] lg:text-[48px] text-[#23262F] text-center b">
          What do the SME's have to say about us?
        </h1>
      </motion.div>
      <div className="xl:w-[920px] m-auto">
        <Swiper
          className="px-9 mt-[50px]"
          modules={[Autoplay, Navigation]}
          navigation={{
            nextEl: ".next",
            prevEl: ".prev",
          }}
          autoplay={{
            delay: 3000,
            disableOnInteraction: false,
            pauseOnMouseEnter: true,
          }}
          spaceBetween={50}
          slidesPerView={2}
          onSlideChange={() => console.log("slide change")}
          onSwiper={(swiper) => console.log(swiper)}
          loop={true}
          breakpoints={{
            0: {
              slidesPerView: 1,
            },
            768: {
              slidesPerView: 2,
            },
          }}
        >
          {feedback.map((f) => (
            <SwiperSlide className="w-full m-auto">
              <div className="flex flex-col justify-between w-auto h-[410px] lg:w-[407px] lg:h-[527px] bg-[#fff] rounded-[20px] p-5 lg:p-10">
                <motion.div
                  initial={{ y: -50, opacity: 0 }}
                  whileInView={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.2, ease: "linear", duration: 0.5 }}
                >
                  <h1 className="text-[14px] lg:text-[16px] text-[#000] s">
                    {f.feedback}
                  </h1>
                </motion.div>
                <div className="flex items-center gap-4">
                  <motion.div
                    initial={{ x: 50, opacity: 0 }}
                    whileInView={{ x: 0, opacity: 1 }}
                    transition={{ ease: "linear", duration: 0.5 }}
                  >
                    <img
                      src={f.profile}
                      alt="profileImg"
                      className="w-[38px] lg:w-[59px]"
                    />
                  </motion.div>
                  <motion.div
                    className="flex flex-col justify-between"
                    initial={{ x: -10, opacity: 0 }}
                    whileInView={{ x: 0, opacity: 1 }}
                    transition={{ delay: 0.6, ease: "linear", duration: 0.5 }}
                  >
                    <h2 className="text-[11px] lg:text-[13px] text-[#000] b">
                      {f.name}
                    </h2>
                    <h3 className="text-[#C2C2C2] text-[10px] lg:text-[12px] r">
                      {f.country}
                    </h3>
                  </motion.div>
                </div>
              </div>
            </SwiperSlide>
          ))}
        </Swiper>
      </div>
      <motion.div
        className="flex items-center gap-[26px] justify-center mt-[38px]"
        initial={{ y: 30, opacity: 0 }}
        whileInView={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.2, ease: "linear", duration: 0.5 }}
      >
        <img
          src="img/arrow.png"
          alt="arrowImage"
          className="-rotate-180 w-[24px] h-[24px] prev cursor-pointer"
        />
        <img
          src="img/arrow.png"
          alt="arrowImage"
          className="w-[24px] h-[24px] next cursor-pointer"
        />
      </motion.div>
    </div>
  );
};
