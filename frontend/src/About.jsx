import React from "react";
import blog from "../src/Image/blog1.png";
import ScrollAnimation from 'react-animate-on-scroll';

const About =()=>{
  return(
    <>
<body className="aboutbody">
<section className="section2" id="AboutUs">
		<table>
		<tr>
		<td>
	
		<ScrollAnimation animateIn="fadeInLeft">
		<img className="aboutimg" src={blog} alt="AboutUs"/>
	 </ScrollAnimation>
		</td>

		<ScrollAnimation animateIn="fadeInRight">
		<td>
		<span className="title">WHAT IS IEM AI WRITER?</span>
		{/* <span className="sub-title">It is your AI-Powered Assistant to write plagiarism-free and grammatically correct content for your website, Articles, Blogs, and so on.</span> */}
		<p className="sub-title1">It is your AI-Powered Assistant to write plagiarism-free and grammatically correct content for your website, Articles, Blogs, and so on.</p>
		<p className="sub-title2">Disclaimer :If you are using our service to generate an AI blog, Please refine it a little bit.</p>
    {/* <a href="" className="btn2">Know More</a> */}
		</td>
		</ScrollAnimation>
		</tr>
		</table>
		</section>
		</body>
    </>
  )
}

export default About;