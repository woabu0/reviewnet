import React from "react";
import InputField from "../Input/InputField";
import Button from "../Input/Button";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";

export const Login = () => {
  return (
    <div id="login">
      <div className="h-[90vh] bg">
        <div className="m-auto px-5 pt-5 md:pt-11 lg:px-9 xl:px-28 2xl:w-[1366px] py-[100px]">
          <motion.div
            className="flex justify-between items-center"
            initial={{ y: -30 }}
            whileInView={{ y: 0 }}
            transition={{ ease: "linear", duration: 0.5 }}
          >
            <Link to="/">
              <img
                src="img/logo.gif"
                alt="logo"
                className="w-[79px] h-[29px] lg:w-[144px] lg:h-[54px]"
              />
            </Link>
            <div className="text-[8px] flex items-center gap-2 lg:gap-[20px] lg:text-[18px] xl:pr-5 m">
              <Link
                to="/demopage"
                className="cursor-pointer w-[73px] h-[24px] lg:w-[180px] lg:h-[45px] border-[#FAFAFA] border-[1px] rounded-[8px] flex items-center justify-center hover:bg-[#FAFAFA] hover:text-[#0049FF] transition-all"
              >
                Book a Demo
              </Link>
            </div>
          </motion.div>
          <div className="absolute top-1/2 left-1/2 translate-x-[-50%] translate-y-[-50%] w-[280px] md:w-[400px] xl:w-[350px] 2xl:w-[400px] m-auto bg-[#FAFAFA] shadow-md p-5 md:p-10 xl:p-5 2xl:p-10 xl:px-10">
            <h1 className="text-center text-[35px] text-[#1A1A1A] b">Login</h1>
            <div className="mt-4 xl:mt-0 2xl:mt-4">
              <InputField
                types="email"
                text="Email Address"
                id="email"
                name="user_email"
                placeholder="e.g., john.doe@example.com"
              />
              <InputField
                types="text"
                text="Username"
                id="username"
                name="user_name"
                placeholder="e.g., john.doe"
              />
              <InputField
                types="password"
                text="Password"
                id="password"
                name="user_password"
                placeholder="e.g., ********"
              />
            </div>
            <p className="my-4 xl:my-2 2xl:my-4 text-[#1a1a1a] text-center text-[14px] m">
              Forgot your password?
            </p>
            <Button title="Login" />
            <p className="my-4 xl:my-2 2xl:my-4 text-[#1a1a1a] text-center text-[14px] r">
              New to our platform?{" "}
              <Link to="/demopage" className="text-[#3772FF] cursor-pointer">
                Get an account now!
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
