const { paginate } = require('gatsby-awesome-pagination')
const { createFilePath } = require(`gatsby-source-filesystem`)
const { forEach, uniq, filter, not, isNil, flatMap } = require('rambdax')
const path = require('path')
const { toKebabCase } = require('./src/helpers')

const pageTemplate = path.resolve(`./src/templates/page.js`)
const postTemplate = path.resolve(`./src/templates/post.js`)
const blogIndexTemplate = path.resolve(`./src/templates/blogIndex.js`)
const tagsTemplate = path.resolve(`./src/templates/tags.js`)

exports.createPages = async ({ actions, graphql, getNodes }) => {
  const { createPage } = actions

  const allMarkdown = await graphql(`
    {
      allMdx(sort: { fields: [frontmatter___date], order: ASC }, limit: 1000) {
        edges {
          node {
            frontmatter {
              path
              title
              tags
            }
            fields {
              collection
            }
            fileAbsolutePath
          }
        }
      }
      site {
        siteMetadata {
          postsPerPage
        }
      }
    }
  `)

  console.log(allMarkdown)

  const {
    allMdx: { edges: markdownPages },
    site: { siteMetadata },
  } = allMarkdown.data

  const filterBy = collection => mp => mp.node.fields.collection === collection
  const pages = markdownPages.filter(filterBy('pages'))
  const posts = markdownPages.filter(filterBy('posts'))

  // Create posts index with pagination
  paginate({
    createPage,
    items: posts,
    component: blogIndexTemplate,
    itemsPerPage: siteMetadata.postsPerPage,
    pathPrefix: '/blog',
  })

  // Create tag pages
  const tags = filter(
    tag => not(isNil(tag)),
    uniq(flatMap(post => post.node.frontmatter.tags, posts)),
  )

  forEach(tag => {
    const postsWithTag = posts.filter(
      post =>
        post.node.frontmatter.tags &&
        post.node.frontmatter.tags.indexOf(tag) !== -1,
    )

    paginate({
      createPage,
      items: postsWithTag,
      component: tagsTemplate,
      itemsPerPage: siteMetadata.postsPerPage,
      pathPrefix: `/tag/${toKebabCase(tag)}`,
      context: {
        tag,
      },
    })
  }, tags)

  forEach(({ node }, index) => {
    const previous = index === 0 ? null : posts[index - 1].node
    const next = index === posts.length - 1 ? null : posts[index + 1].node

    createPage({
      path: node.frontmatter.path,
      component: postTemplate,
      context: {
        next,
        previous,
      },
    })
  }, posts)

  forEach(({ node }, index) => {
    createPage({
      path: node.frontmatter.path,
      component: pageTemplate,
    })
  }, pages)

  return {
    sortedPages: posts,
    tags,
  }
}

exports.sourceNodes = ({ actions }) => {
  const { createTypes } = actions
  const typeDefs = `
    type Mdx implements Node {
      frontmatter: Frontmatter!
    }

    type Frontmatter {
      title: String!
      author: String
      date: Date! @dateformat
      path: String!
      tags: [String!]
      excerpt: String
      coverImage: File @fileByRelativePath
    }
  `
  createTypes(typeDefs)
}

exports.onCreateNode = ({ node, getNode, actions }) => {
  if (node.internal.type === `Mdx`) {
    const { createNodeField } = actions
    const parent = getNode(node.parent)
    createNodeField({
      node,
      name: 'collection',
      value: parent.sourceInstanceName,
    })
  }
}
