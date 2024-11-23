import React from "react";
import { motion } from "framer-motion";
import Aicards from "./Aicards";

export const Ai = () => {
  return (
    <div
      id="diseases"
      className="mt-[70px] px-[10px] lg:px-9 m-auto 2xl:w-[1366px] bg-[#b7e9f7] text-black text-center py-1 rounded-[30px] aibg"
    >
      <div className="my-10">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          whileInView={{ scale: 1, opacity: 1 }}
          transition={{ ease: "linear", duration: 0.3 }}
          className="my-10 block xl:hidden"
        >
          <h1 className="mb-5 b text-[25px] leading-tight">
            AI for Mitigating the <br /> risk of Ocular Diseases
          </h1>
          <p className="text-[14px] m-auto mt-5 r">
            Harnessing the power of artificial intelligence, Amoha.ai's
            proprietary cloud-based platform is streamlining ocular health. With
            highly accurate early detection and diagnoses, it grants both
            patients and providers invaluable time for intervention, preserving
            vision and preventing severe impairment. By embracing proactive and
            predictive AI approaches, we're at the forefront of advancing eye
            care, ensuring a brighter future for all.
          </p>
        </motion.div>
        <div className="flex flex-col xl:flex-row gap-2 justify-between relative">
          <motion.div
            initial={{
              x: "var(--x-from)",
              y: "var(--y-from)",
              scale: 0.8,
              opacity: 0,
            }}
            whileInView={{
              x: "var(--x-to)",
              y: "var(--y-to)",
              scale: 1,
              opacity: 1,
            }}
            transition={{ ease: "linear", duration: 0.6, delay: 0.2 }}
            className="bg-[#D9FFDB] w-auto xl:w-[400px] h-[280px] rounded-[30px] pt-5 p-2 xl:p-5 [--x-from:0px] [--x-to:0px] xl:[--x-from:100px] xl:[--x-to:0px] [--y-from:-30px] [--y-to:0px] xl:[--y-from:100px] xl:[--y-to:0px] expand"
          >
            <Aicards
              title="Unparalleled Predictive Precision"
              details="Through machine learning algorithms and advanced pattern recognition, our AI is trained on vast datasets of ocular imagery and health records. This enables it to detect early signs of eye diseases such as macular degeneration, diabetic retinopathy, and glaucoma, even before symptoms become noticeable to patients, increasing the effectiveness of preventive therapies."
            />
          </motion.div>
          <motion.div
            initial={{
              x: "var(--x-from)",
              y: "var(--y-from)",
              scale: 0.8,
              opacity: 0,
            }}
            whileInView={{
              x: "var(--x-to)",
              y: "var(--y-to)",
              scale: 1,
              opacity: 1,
            }}
            transition={{ ease: "linear", duration: 0.6, delay: 0.3 }}
            className="bg-[#F2E6FF] w-auto xl:w-[400px] h-[280px] rounded-[30px] pt-5 p-2 xl:p-5 [--x-from:0px] [--x-to:0px] xl:[--x-from:-100px] xl:[--x-to:0px] [--y-from:-30px] [--y-to:0px] xl:[--y-from:100px] xl:[--y-to:0px] expand"
          >
            <Aicards
              title="Transform Data into Insights"
              details="Not every eye is the same. We use AI to derive predictive insights from a range of patient-specific data, including health metrics, lifestyle factors, and genetic data. By integrating these multi-dimensional insights, we facilitate personalized treatment plans to minimize the risk of ocular diseases on an individual level."
            />
          </motion.div>
        </div>
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          whileInView={{ scale: 1, opacity: 1 }}
          transition={{ ease: "linear", duration: 0.3 }}
          className="my-10 hidden xl:block"
        >
          <h1 className="mb-5 lg:text-[50px] xl:mb-0 b text-[22px] leading-tight">
            AI for Mitigating the <br /> risk of Ocular Diseases
          </h1>
          <p className="text-[14px] lg:text-[16px] m-auto mt-5 w-[80%] r">
            Harnessing the power of artificial intelligence, Amoha.ai's
            proprietary cloud-based platform is streamlining ocular health. With
            highly accurate early detection and diagnoses, it grants both
            patients and providers invaluable time for intervention, preserving
            vision and preventing severe impairment. By embracing proactive and
            predictive AI approaches, we're at the forefront of advancing eye
            care, ensuring a brighter future for all.
          </p>
        </motion.div>
        <div className="flex flex-col xl:flex-row gap-2 mt-2 justify-between">
          <motion.div
            initial={{
              x: "var(--x-from)",
              y: "var(--y-from)",
              scale: 0.8,
              opacity: 0,
            }}
            whileInView={{
              x: "var(--x-to)",
              y: "var(--y-to)",
              scale: 1,
              opacity: 1,
            }}
            transition={{ ease: "linear", duration: 0.6, delay: 0.5 }}
            className="bg-[#E6F7FF] w-auto xl:w-[400px] h-[280px] rounded-[30px] pt-5 p-2 xl:p-5 [--x-from:0px] [--x-to:0px] xl:[--x-from:100px] xl:[--x-to:0px] [--y-from:-30px] [--y-to:0px] xl:[--y-from:-100px] xl:[--y-to:0px] expand"
          >
            <Aicards
              title="Optimizing Clinical Decisions"
              details="Our models support healthcare providers by offering clinical decision support and reducing diagnostic errors. The AI-based toolkits help practitioners double-check their diagnoses, improve their workload, and prioritize patients who need more urgent intervention."
            />
          </motion.div>
          <motion.div
            initial={{
              x: "var(--x-from)",
              y: "var(--y-from)",
              scale: 0.8,
              opacity: 0,
            }}
            whileInView={{
              x: "var(--x-to)",
              y: "var(--y-to)",
              scale: 1,
              opacity: 1,
            }}
            transition={{ ease: "linear", duration: 0.6, delay: 0.4 }}
            className="bg-[#F2F2F2] w-auto xl:w-[400px] h-[280px] rounded-[30px] pt-5 p-2 xl:p-5 [--x-from:0px] [--x-to:0px] xl:[--x-from:-100px] xl:[--x-to:0px] [--y-from:-30px] [--y-to:0px] xl:[--y-from:-100px] xl:[--y-to:0px] expand"
          >
            <Aicards
              title="Ensemble of Security and Privacy"
              details="We take data security and privacy very seriously. Our systems are designed in accordance with global standards for health data privacy to ensure that your sensitive information is treated with the utmost respect and protection."
            />
          </motion.div>
        </div>
      </div>
    </div>
  );
};
