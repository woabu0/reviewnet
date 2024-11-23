import React from "react";
import { BookPage } from "../BookPage/BookPage";
import { UniversalFooter } from "../Footer/UniversalFooter";
import { Helmet, HelmetData } from "react-helmet-async";

export const DemoPage = () => {
  const helmetData = new HelmetData({});
  return (
    <div className="demopage">
      <Helmet helmetData={helmetData}>
        <title>Amoha.ai - Book a Demo</title>
        <link rel="canonical" href="https://amoha.ai/demopage" />
      </Helmet>
      <BookPage />
      <UniversalFooter />
    </div>
  );
};
