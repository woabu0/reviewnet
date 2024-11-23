import React from "react";
import { BrowserRouter as Main, Route, Routes } from "react-router-dom";
import { HomePage } from "./components/Pages/HomePage";
import { LoginPage } from "./components/Pages/LoginPage";
import { DemoPage } from "./components/Pages/DemoPage";
import { ErrorPage } from "./components/Pages/ErrorPage";
import { Helmet, HelmetProvider } from "react-helmet-async";

function App() {
  return (
    <HelmetProvider>
      <Helmet>
        <title>Amoha.ai</title>
        <meta
          name="description"
          content="Amoha.ai is an innovative Software-as-a-Service (SaaS) MedTech platform designed to revolutionize the early detection and management of various ocular conditions such as diabetic retinopathy, glaucoma, AMD, cataracts, etc., which, if unchecked, can lead to significant vision loss or even blindness."
        />
      </Helmet>
      <Main>
        <Routes>
          <Route exact path="/" element={<HomePage />} />
          <Route exact path="/loginpage" element={<LoginPage />} />
          <Route exact path="/demopage" element={<DemoPage />} />
          <Route path="*" element={<ErrorPage />} />
        </Routes>
      </Main>
    </HelmetProvider>
  );
}

export default App;
