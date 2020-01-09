import React from "react";
import PropTypes from "prop-types";
import SEO from "../components/seo";
import Layout from "../components/layout";

import style from "../styles/index.module.css";

const Index = () => {
  return (
    <>
      <SEO />
      <Layout footer={false}>
        <div className={`container mt-5 mt-md-0 ${style.textCenter} ${style.textMdLeft}`}>
          <div
            className={`row mb-5 ${style.justifyContentAround} ${style.flexMdNowrap}`}
          >
            <div className="col col-md-8 col-12 order-md-1 order-2">
              <h1 className={style.heading}>Hello, I'm Tom.</h1>
              <p className={style.summary}>
                I am a data scientist and mathematician. I'm currently R&D Lead
                at <a href="https://faculty.ai">Faculty</a>
              </p>
            </div>
            <div className="col col-auto mb-4 mb-md-0">
              <img
                src="/assets/me.jpg"
                width={200}
                style={{ borderRadius: 8 }}
              />
            </div>
          </div>
          <p>
            You've found my website. It has some information about me, and
            things I've worked on both professionally and in my spare time. I've
            also aspirationally set up a blog that I occasionally get around to
            writing things for. I'm interested in many things including, but not
            limited to Bayesian inference, machine learning, web development,
            and programming with Python.
          </p>
        </div>
      </Layout>
    </>
  );
};

Index.propTypes = {
  data: PropTypes.object.isRequired,
  pageContext: PropTypes.shape({
    nextPagePath: PropTypes.string,
    previousPagePath: PropTypes.string,
  }),
};

export default Index;
