import React, { useRef } from "react";
import { motion } from "framer-motion";
import emailjs from "@emailjs/browser";

export const Footer = () => {
  const form = useRef();

  const sendEmail = (e) => {
    e.preventDefault();

    emailjs
      .sendForm(
        "service_7tgyrxh",
        "template_m41upyo",
        form.current,
        "X2BgLpJsY18HGn10P"
      )
      .then(
        (result) => {
          console.log(result.text);
          alert("Successfully send");
          document.getElementById("subscriber").value = "";
        },
        (error) => {
          console.log(error.text);
        }
      );
  };
  return (
    <div id="footer" className="bg-black mt-[50px] xl:mt-[125px]">
      <div className="w-full 2xl:w-[1366px] m-auto pt-[80px] px-9">
        <div className="flex flex-col justify-between xl:flex-row gap-[50px]">
          <motion.div
            initial={{ x: -50 }}
            whileInView={{ x: 0 }}
            transition={{ ease: "linear", duration: 0.5 }}
          >
            <img
              src="img/logo-footer.gif"
              alt="footerLogo"
              className="w-[110px] h-[33px]"
            />
          </motion.div>
          <motion.div
            className="flex flex-col gap-[25px]"
            initial={{ x: -50, opacity: 0 }}
            whileInView={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.6, ease: "linear", duration: 0.5 }}
          >
            <h3 className="text-[16px] m">Subscribe to our newsletter</h3>
            <p className="text-[14px] w-[90%] r">
              Stay informed about the latest features, product enhancements,
              news, and more.
            </p>
            <form
              ref={form}
              onSubmit={sendEmail}
              autoComplete="off"
              className="r h-[48px] w-full border-[#fff] border-[2px] rounded-[90px] focus:outline-none text-[14px] text-white flex items-center"
            >
              <input
                type="email"
                id="subscriber"
                name="subscriber"
                placeholder="Enter your email"
                className="r bg-[#000] h-full mx-[16px] rounded-[10px] w-full focus:outline-none text-[14px]"
              />
              <button
                type="submit"
                className="w-[32px] h-[32px] mr-[8px] cursor-pointer"
              >
                <img src="img/footer-arrow.png" alt="footerArrow" />
              </button>
            </form>
          </motion.div>
        </div>
        <div className="mt-[100px] w-full">
          <motion.div
            className="h-[1px] w-full bg-white m-auto"
            initial={{ width: 0 }}
            whileInView={{ width: "100%" }}
            transition={{ delay: 0.3, ease: "linear", duration: 0.8 }}
          ></motion.div>
          <motion.div
            className="flex flex-col lg:flex-row justify-between py-[31px]"
            initial={{ y: -30, opacity: 0 }}
            whileInView={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.4, ease: "linear", duration: 0.2 }}
          >
            <h6 className="text-[12px] r text-[#777E90]">
              Copyright © 2024 amoha.ai
            </h6>
            <h6 className="text-[12px]">
              Privacy Policy
            </h6>
          </motion.div>
        </div>
      </div>
    </div>
  );
};
