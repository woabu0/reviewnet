import React from "react";
import dis from "./diseasesData.json";
import { motion } from "framer-motion";

export const Diseases = () => {
  return (
    <div id="diseases" className="mt-[70px] px-[10px] lg:px-9">
      <div className="m-auto 2xl:w-[1366px]">
        <motion.div
          className="mt-[50px] lg:mt-[100px]"
          initial={{ y: -30, opacity: 0 }}
          whileInView={{ y: 0, opacity: 1 }}
          transition={{ ease: "linear", duration: 0.5 }}
        >
          <h1 className="text-[24px] lg:text-[48px] text-[#23262F] text-center b md:w-[60%] m-auto">
            Ever imagined the world through the lens of these conditions?
          </h1>
        </motion.div>
        <div className="mt-[100px]">
          {dis.map((d) => (
            <div
              className="flex flex-col-reverse gap-5 md:flex-row items-center justify-between w-auto m-auto text-[#101010]"
              id={d.id}
            >
              <motion.div
                className="w-full md:w-1/2"
                initial={{ y: -30, opacity: 0 }}
                whileInView={{ y: 0, opacity: 1 }}
                transition={{ ease: "linear", duration: 0.5, delay: 0.1 }}
              >
                <iframe
                  className="mb-5 rounded-[16px] w-full h-[275px] lg:h-[375px]"
                  src={d.video}
                  title="@AmohaAI"
                  frameborder="0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                  allowFullScreen
                ></iframe>
              </motion.div>
              <motion.div
                className="md:w-1/2"
                initial={{ y: -30, opacity: 0 }}
                whileInView={{ y: 0, opacity: 1 }}
                transition={{ ease: "linear", duration: 0.5, delay: 0.3 }}
              >
                <h1 className="text-[35px] lg:text-[40px] b">{d.title}</h1>
                <p className="mt-[25px] text-[14px] lg:text-[18px] m">
                  {d.detail}
                </p>
              </motion.div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
