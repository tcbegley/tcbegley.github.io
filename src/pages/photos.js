import React from "react";
import { graphql } from "gatsby";
import { Carousel } from "react-responsive-carousel";
import "react-responsive-carousel/lib/styles/carousel.min.css";
import SEO from "../components/seo";
import Layout from "../components/layout";
import style from "../styles/post.module.css";
import "../styles/photos.css";

export default ({ data }) => (
  <>
    <SEO title={"Photos"} />
    <Layout>
      <div className={style.post}>
        <div className={style.postContent}>
          <h1 className={style.title}>Photos</h1>
          <p>
            I'm a keen amateur photographer. On this page you can find a few of
            the photos I've taken over the years.
          </p>
          <Carousel dynamicHeight showIndicators={false}>
            {data.allFlickrPhoto.edges.map(edge => (
              <div style={{ height: "500px" }}>
                <img
                  src={edge.node.url_c}
                  style={{ height: "auto", width: "auto" }}
                />
                <p className="legend">{edge.node.description}</p>
              </div>
            ))}
          </Carousel>
          All of these photos are also available on my{" "}
          <a href="https://flickr.com/photos/149210668@N06/">Flickr</a> account.
        </div>
      </div>
    </Layout>
  </>
);

export const query = graphql`
  query {
    allFlickrPhoto {
      edges {
        node {
          id
          url_c
          description
        }
      }
    }
  }
`;
