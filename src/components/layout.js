import React from "react";
import PropTypes from "prop-types";
import { useStaticQuery, graphql } from "gatsby";

import Header from "./header";
import Footer from "./footer";

import "../styles/layout.css";

const Layout = ({ children, column }) => {
  const data = useStaticQuery(graphql`
    query SiteTitleQuery {
      site {
        siteMetadata {
          title
          logo {
            src
            alt
          }
          logoText
          defaultTheme
          copyrights
          mainMenu {
            title
            path
          }
          showMenuItems
          menuMoreText
        }
      }
    }
  `);
  const {
    title,
    logo,
    logoText,
    defaultTheme,
    mainMenu,
    showMenuItems,
    menuMoreText,
    copyrights,
  } = data.site.siteMetadata;

  const flexColumnStyle = {
    flexDirection: "column",
    justifyContent: "flex-start",
  };

  return (
    <div className="page-container">
      <Header
        siteTitle={title}
        siteLogo={logo}
        logoText={logoText}
        defaultTheme={defaultTheme}
        mainMenu={mainMenu}
        mainMenuItems={showMenuItems}
        menuMoreText={menuMoreText}
      />
      <div className="content" style={column ? flexColumnStyle : null}>
        {children}
      </div>
      <Footer copyrights={copyrights} />
    </div>
  );
};

Layout.propTypes = {
  children: PropTypes.node.isRequired,
  footer: PropTypes.bool,
};

Layout.defaultProps = {
  footer: true,
};

export default Layout;
