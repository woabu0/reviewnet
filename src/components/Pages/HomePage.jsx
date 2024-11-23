import React from "react";
import { Home } from "../Home/Home";
import { Diseases } from "../Diseases/Diseases";
import { Ai } from "../Ai/Ai";
import { Train } from "../Train/Train";
import { Feedback } from "../Feedback/Feedback";
import { Footer } from "../Footer/Footer";
import { Contact } from "../Contact/Contact";

export const HomePage = () => {
  return (
    <div>
      <Home />
      <Diseases />
      <Ai />
      <Train />
      <Feedback />
      <Contact />
      <Footer />
    </div>
  );
};
