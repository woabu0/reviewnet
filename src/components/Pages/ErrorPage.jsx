import React from "react";
import { Helmet, HelmetData } from "react-helmet-async";

export const ErrorPage = () => {
  const helmetData = new HelmetData({});
  return (
    <div className="text-[100px] text-center b">
      <Helmet helmetData={helmetData}>
        <title>Amoha.ai - Error</title>
        <link rel="canonical" href="https://www.tacobell.com/" />
      </Helmet>
      404 Pgae Not found
    </div>
  );
};
