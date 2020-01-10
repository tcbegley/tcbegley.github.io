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
        <div className={style.indexContainer}>
          <div className={style.indexRow}>
            <div className={style.textCol}>
              <h1 className={style.heading}>Hello, I'm Tom.</h1>
              <p className={style.summary}>
                I am a data scientist and mathematician. I'm currently R&D Lead
                at <a href="https://faculty.ai">Faculty</a>
              </p>
            </div>
            <div className={style.imgCol}>
              <img
                src="/assets/me.jpg"
                width={200}
                style={{ borderRadius: 8 }}
              />
            </div>
          </div>
          <p className={style.text}>
            You've found my website. It has some information about me, and
            things I've worked on both professionally and in my spare time. I've
            also aspirationally set up a blog that I occasionally get around to
            writing things for. I'm interested in many things including, but not
            limited to Bayesian inference, machine learning, web development,
            and programming with Python.
          </p>
          <span className={style.social}>
            <a href="https://github.com/tcbegley">
              <i class="fab fa-2x fa-github"></i>
            </a>
            <a href="https://linkedin.com/in/tcbegley">
              <i class="fab fa-2x fa-linkedin"></i>
            </a>
            <a href="https://twitter.com/tcbegley">
              <i class="fab fa-2x fa-twitter"></i>
            </a>
          </span>
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