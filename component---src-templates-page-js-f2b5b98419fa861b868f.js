(window.webpackJsonp=window.webpackJsonp||[]).push([[10],{"+pmV":function(t,e,n){t.exports={post:"post-module--post--28Mq2",title:"post-module--title--3XBo2",coverImage:"post-module--coverImage--1GM7V",meta:"post-module--meta--3YtjE",tags:"post-module--tags--3RbqF",tag:"post-module--tag--16U9p",darkTheme:"post-module--dark-theme--DXAZf",readMore:"post-module--readMore--3zWML",postContent:"post-module--postContent--1bfnt"}},"2klF":function(t,e){t.exports=function(){if("undefined"==typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"==typeof Proxy)return!0;try{return Date.prototype.toString.call(Reflect.construct(Date,[],(function(){}))),!0}catch(t){return!1}}},"5l6m":function(t,e,n){"use strict";var r=n("+uX7"),a=n("m/aQ"),o=n("17+C"),c=n("WD+B"),i=n("gQbX"),s=n("4jnk"),u=n("5Cvy"),l=n("mh2x"),p=Math.max,f=Math.min,m=Math.floor,v=/\$([$&'`]|\d\d?|<[^>]*>)/g,g=/\$([$&'`]|\d\d?)/g;r("replace",2,(function(t,e,n,r){var d=r.REGEXP_REPLACE_SUBSTITUTES_UNDEFINED_CAPTURE,b=r.REPLACE_KEEPS_$0,h=d?"$":"$0";return[function(n,r){var a=s(this),o=null==n?void 0:n[t];return void 0!==o?o.call(n,a,r):e.call(String(a),n,r)},function(t,r){if(!d&&b||"string"==typeof r&&-1===r.indexOf(h)){var o=n(e,t,this,r);if(o.done)return o.value}var s=a(t),m=String(this),v="function"==typeof r;v||(r=String(r));var g=s.global;if(g){var x=s.unicode;s.lastIndex=0}for(var E=[];;){var P=l(s,m);if(null===P)break;if(E.push(P),!g)break;""===String(P[0])&&(s.lastIndex=u(m,c(s.lastIndex),x))}for(var j,O="",k=0,w=0;w<E.length;w++){P=E[w];for(var R=String(P[0]),N=p(f(i(P.index),m.length),0),S=[],I=1;I<P.length;I++)S.push(void 0===(j=P[I])?j:String(j));var M=P.groups;if(v){var A=[R].concat(S,N,m);void 0!==M&&A.push(M);var C=String(r.apply(void 0,A))}else C=y(R,m,N,S,M,r);N>=k&&(O+=m.slice(k,N)+C,k=N+R.length)}return O+m.slice(k)}];function y(t,n,r,a,c,i){var s=r+t.length,u=a.length,l=g;return void 0!==c&&(c=o(c),l=v),e.call(i,l,(function(e,o){var i;switch(o.charAt(0)){case"$":return"$";case"&":return t;case"`":return n.slice(0,r);case"'":return n.slice(s);case"<":i=c[o.slice(1,-1)];break;default:var l=+o;if(0===l)return e;if(l>u){var p=m(l/10);return 0===p?e:p<=u?void 0===a[p-1]?o.charAt(1):a[p-1]+o.charAt(1):e}i=a[l-1]}return void 0===i?"":i}))}}))},"6cYQ":function(t,e,n){"use strict";var r=n("q1tI"),a=n.n(r),o=n("17x9"),c=n.n(o),i=n("Wbzz"),s=n("zHTP"),u=n.n(s),l=function(t){var e=t.nextPath,n=t.previousPath,r=t.nextLabel,o=t.previousLabel;return n||e?a.a.createElement("div",{className:u.a.navigation},n&&a.a.createElement("span",{className:u.a.button},a.a.createElement(i.Link,{to:n},a.a.createElement("span",{className:u.a.iconPrev},"←"),a.a.createElement("span",{className:u.a.buttonText},o))),e&&a.a.createElement("span",{className:u.a.button},a.a.createElement(i.Link,{to:e},a.a.createElement("span",{className:u.a.buttonText},r),a.a.createElement("span",{className:u.a.iconNext},"→")))):null};l.propTypes={nextPath:c.a.string,previousPath:c.a.string,nextLabel:c.a.string,previousLabel:c.a.string},e.a=l},"A2+M":function(t,e,n){var r=n("X8hv");t.exports={MDXRenderer:r}},Ck4i:function(t,e,n){var r=n("Q83E"),a=n("2klF");function o(e,n,c){return a()?t.exports=o=Reflect.construct:t.exports=o=function(t,e,n){var a=[null];a.push.apply(a,e);var o=new(Function.bind.apply(t,a));return n&&r(o,n.prototype),o},o.apply(null,arguments)}t.exports=o},Q83E:function(t,e){function n(e,r){return t.exports=n=Object.setPrototypeOf||function(t,e){return t.__proto__=e,t},n(e,r)}t.exports=n},R7tm:function(t,e,n){var r=n("qHws"),a=n("gC2u"),o=n("dQcQ"),c=n("m7BV");t.exports=function(t){return r(t)||a(t)||o(t)||c()}},X8hv:function(t,e,n){var r=n("Ck4i"),a=n("R7tm"),o=n("0jh0"),c=n("uDP2");function i(t,e){var n=Object.keys(t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(t);e&&(r=r.filter((function(e){return Object.getOwnPropertyDescriptor(t,e).enumerable}))),n.push.apply(n,r)}return n}function s(t){for(var e=1;e<arguments.length;e++){var n=null!=arguments[e]?arguments[e]:{};e%2?i(Object(n),!0).forEach((function(e){o(t,e,n[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(t,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(e){Object.defineProperty(t,e,Object.getOwnPropertyDescriptor(n,e))}))}return t}var u=n("q1tI"),l=n("7ljp").mdx,p=n("BfwJ").useMDXScope;t.exports=function(t){var e=t.scope,n=t.children,o=c(t,["scope","children"]),i=p(e),f=u.useMemo((function(){if(!n)return null;var t=s({React:u,mdx:l},i),e=Object.keys(t),o=e.map((function(e){return t[e]}));return r(Function,["_fn"].concat(a(e),[""+n])).apply(void 0,[{}].concat(a(o)))}),[n,e]);return u.createElement(f,s({},o))}},dQcQ:function(t,e,n){var r=n("hMe3");t.exports=function(t,e){if(t){if("string"==typeof t)return r(t,e);var n=Object.prototype.toString.call(t).slice(8,-1);return"Object"===n&&t.constructor&&(n=t.constructor.name),"Map"===n||"Set"===n?Array.from(t):"Arguments"===n||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)?r(t,e):void 0}}},gC2u:function(t,e){t.exports=function(t){if("undefined"!=typeof Symbol&&Symbol.iterator in Object(t))return Array.from(t)}},hMe3:function(t,e){t.exports=function(t,e){(null==e||e>t.length)&&(e=t.length);for(var n=0,r=new Array(e);n<e;n++)r[n]=t[n];return r}},kAaw:function(t,e,n){var r=n("IvzW"),a=n("REpN"),o=n("ZRnM"),c=n("nynY"),i=n("jekk").f,s=n("zpoX").f,u=n("iwAE"),l=n("7npg"),p=n("zJsW"),f=n("+7hJ"),m=n("JhOX"),v=n("E9J1").set,g=n("43WK"),d=n("QD2z")("match"),b=a.RegExp,h=b.prototype,y=/a/g,x=/a/g,E=new b(y)!==y,P=p.UNSUPPORTED_Y;if(r&&o("RegExp",!E||P||m((function(){return x[d]=!1,b(y)!=y||b(x)==x||"/a/i"!=b(y,"i")})))){for(var j=function(t,e){var n,r=this instanceof j,a=u(t),o=void 0===e;if(!r&&a&&t.constructor===j&&o)return t;E?a&&!o&&(t=t.source):t instanceof j&&(o&&(e=l.call(t)),t=t.source),P&&(n=!!e&&e.indexOf("y")>-1)&&(e=e.replace(/y/g,""));var i=c(E?new b(t,e):b(t,e),r?this:h,j);return P&&n&&v(i,{sticky:n}),i},O=function(t){t in j||i(j,t,{configurable:!0,get:function(){return b[t]},set:function(e){b[t]=e}})},k=s(b),w=0;k.length>w;)O(k[w++]);h.constructor=j,j.prototype=h,f(a,"RegExp",j)}g("RegExp")},m7BV:function(t,e){t.exports=function(){throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}},qHws:function(t,e,n){var r=n("hMe3");t.exports=function(t){if(Array.isArray(t))return r(t)}},rgsX:function(t,e,n){"use strict";var r=n("q1tI"),a=n.n(r),o=n("17x9"),c=n.n(o),i=n("Wbzz"),s=n("9eSz"),u=n.n(s),l=n("A2+M"),p=n("6cYQ"),f=n("zpb6"),m=n("+pmV"),v=n.n(m),g=function(t){var e=t.title,n=t.date,r=t.path,o=t.coverImage,c=t.author,s=t.excerpt,m=t.tags,g=t.body,d=t.previousPost,b=t.nextPost,h=d&&d.frontmatter.path,y=d&&d.frontmatter.title,x=b&&b.frontmatter.path,E=b&&b.frontmatter.title;return a.a.createElement("div",{className:v.a.post},a.a.createElement("div",{className:v.a.postContent},a.a.createElement("h1",{className:v.a.title},s?a.a.createElement(i.Link,{to:r},e):e),a.a.createElement("div",{className:v.a.meta},n," ",c&&a.a.createElement(a.a.Fragment,null,"— Written by ",c),m?a.a.createElement("div",{className:v.a.tags},m.map((function(t){return a.a.createElement(i.Link,{to:"/tag/"+Object(f.toKebabCase)(t)+"/",key:Object(f.toKebabCase)(t)},a.a.createElement("span",{className:v.a.tag},"#",t))}))):null),o&&a.a.createElement(u.a,{fluid:o.childImageSharp.fluid,className:v.a.coverImage}),s?a.a.createElement(a.a.Fragment,null,a.a.createElement("p",null,s),a.a.createElement(i.Link,{to:r,className:v.a.readMore},"Read more →")):a.a.createElement(a.a.Fragment,null,a.a.createElement(l.MDXRenderer,null,g),a.a.createElement(p.a,{previousPath:h,previousLabel:y,nextPath:x,nextLabel:E}))))};g.propTypes={title:c.a.string,date:c.a.string,path:c.a.string,coverImage:c.a.object,author:c.a.string,excerpt:c.a.string,body:c.a.any,tags:c.a.arrayOf(c.a.string),previousPost:c.a.object,nextPost:c.a.object},e.a=g},sweJ:function(t,e,n){"use strict";n.r(e),n.d(e,"pageQuery",(function(){return p}));var r=n("q1tI"),a=n.n(r),o=n("17x9"),c=n.n(o),i=n("vrFN"),s=n("Bl7J"),u=n("rgsX"),l=function(t){var e=t.data.mdx,n=e.frontmatter,r=n.title,o=n.path,c=n.coverImage,l=n.excerpt,p=n.tags,f=e.excerpt,m=e.id,v=e.body;return a.a.createElement(s.a,null,a.a.createElement(i.a,{title:r,description:l||f}),a.a.createElement(u.a,{key:m,title:r,path:o,coverImage:c,body:v,tags:p}))};e.default=l,l.propTypes={data:c.a.object.isRequired,pageContext:c.a.shape({next:c.a.object,previous:c.a.object})};var p="3099823849"},zHTP:function(t,e,n){t.exports={navigation:"navigation-module--navigation--3Zfju",button:"navigation-module--button--28kp3",buttonText:"navigation-module--buttonText--1Xod2",iconNext:"navigation-module--iconNext--3xyJ-",iconPrev:"navigation-module--iconPrev--23mg1"}},zpb6:function(t,e,n){n("kAaw"),n("5l6m"),t.exports.toKebabCase=function(t){return t.replace(new RegExp("(\\s|_|-)+","gmi"),"-")}}}]);
//# sourceMappingURL=component---src-templates-page-js-f2b5b98419fa861b868f.js.map