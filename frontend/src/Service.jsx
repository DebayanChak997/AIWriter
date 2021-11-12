import React from "react";
import { $ }  from 'react-jquery-plugin'
// import web from "../src/Image/web1.jpg";
// import Card from "./Card";
// import Sdata from "./Sdata";
import "./index1.scss";
import ScrollAnimation from 'react-animate-on-scroll';



const Services = (props) => {
  return (
    <>

<body>
<ScrollAnimation animateIn="bounceInRight">
<div id="generic_price_table">   
<section>
       <h2 className="serviceh2"> Service</h2>
        <div className="container">
            
         
            <div className="row">
                <div className="col-md-4">
                
                    <div className="generic_content clearfix">
                        
                      
                        <div className="generic_head_price clearfix">
                        
                           
                            <div className="generic_head_content clearfix">
                            
                                <div className="head_bg"></div>
                                <div className="head">
                                    <span>Basic</span>
                                </div>
                              
                            </div>
                          
                            <div className="generic_price_tag clearfix">	
                                <span className="price">
                                    <span className="sign">$</span>
                                    <span className="currency">99</span>
                                    <span className="cent">.99</span>
                                    <span className="month">/MON</span>
                                </span>
                            </div>
                           
                            
                        </div>                            
                       
                        <div className="generic_feature_list">
                        	<ul>
                            	<li><span>10</span> Writer Coins</li>
                                <li><span>10</span> Emails</li>
                                <li><span></span> Blog Generation</li>
                                <li><span></span> Chat Support</li>
                               
                            </ul>
                        </div>
                       
                        <div className="generic_price_btn clearfix">
                        	<a className="" href="">Buy Now</a>
                        </div>
                        
                        
                    </div>      
                </div>
                
                <div className="col-md-4">
                
             
                    <div className="generic_content active clearfix">
                       
                        <div className="generic_head_price clearfix">
                     
                            <div className="generic_head_content clearfix">
                          
                                <div className="head_bg"></div>
                                <div className="head">
                                    <span>PRO</span>
                                </div>
                            
                                
                            </div>
                        
                            <div className="generic_price_tag clearfix">	
                                <span className="price">
                                    <span className="sign">$</span>
                                    <span className="currency">199</span>
                                    <span className="cent">.99</span>
                                    <span className="month">/MON</span>
                                </span>
                            </div>
                           
                            
                        </div>                            
                        
                        <div className="generic_feature_list">
                        	<ul>
                            	<li><span>20</span> Writer Coins</li>
                                <li><span>20</span> Emails</li>
                                <li><span></span>Blog Generation</li>
                                <li><span></span>Chat Support</li>
                                {/* <li><span>24/7</span> Support</li> */}
                            </ul>
                        </div>
                        
                        <div className="generic_price_btn clearfix">
                        	<a className="" href="">Buy Now</a>
                        </div>
                      
                    </div>
                 
                        
                </div>
                <div className="col-md-4">
               
                    <div className="generic_content clearfix">
                        
                       
                        <div className="generic_head_price clearfix">
                        
                            
                            <div className="generic_head_content clearfix">
                            
                            
                                <div className="head_bg"></div>
                                <div className="head">
                                    <span>PREMIUM</span>
                                </div>
                             
                                
                            </div>
                            
                            <div className="generic_price_tag clearfix">	
                                <span className="price">
                                    <span className="sign">$</span>
                                    <span className="currency">299</span>
                                    <span className="cent">.99</span>
                                    <span className="month">/MON</span>
                                </span>
                            </div>
                         
                            
                        </div>                            
                      
                        <div className="generic_feature_list">
                        	<ul>
                            	<li><span>30</span> Writer Coins</li>
                                <li><span>30</span> Emails</li>
                                <li><span></span> Blog Generation</li>
                                <li><span></span>Chat Support</li>
                                {/* <li><span>24/7</span> Support</li> */}
                            </ul>
                        </div>
                       
                        <div className="generic_price_btn clearfix">
                        	<a className="" href="">Buy Now</a>
                        </div>
                     
                        
                    </div>
                  
                        
                </div>
            </div>	
          
            
        </div>
    </section>             

</div>


    </ScrollAnimation>




  </body>


      {/* <div className="my-5">
        <h1 className="text-center"> Our Services </h1>
      </div>

      <div className="container-fluid mb-5">
        <div className="row ">
          <div className="col-10 mx-auto">
            <div className="row gy-5">
            
              {Sdata.map((val, ind) => {
                return <Card imgsrc={val.imgsrc} title={val.title} />;
              })}
            </div>
          </div>
        </div>
      </div> */}
    </>
  );
};

export default Services;
